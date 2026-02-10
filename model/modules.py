import math
import numpy as np
import einops
import torch
import torch.nn.functional as F
from typing import Optional, Final, Callable, Tuple, Union, Iterable, List

from scipy.signal import get_window
from torch import nn, Tensor
from functools import partial

from utils import as_complex, as_real
from init_norms import InitMagNorm, InitSpecNorm


class ResidualConvBlock(nn.Module):
    def __init__(self, conv_layer_fn, depth, conv_ch, point_wise_type):
        super().__init__()

        self.first_layer = conv_layer_fn

        block_fn = partial(
            Conv2dNormAct,
            in_ch=conv_ch,
            out_ch=conv_ch,
            kernel_size=(1, 3),
            fstride=1,
            bias=False,
            separable=True,
            point_wise_type=point_wise_type,
            activation_layer=nn.Identity
        )

        self.res_blocks = nn.ModuleList([block_fn() for _ in range(depth - 1)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.first_layer(x)
        for block in self.res_blocks:
            residual = out
            out = block(out)
            out = out + residual
            out = self.relu(out)
        return out


def make_deep_conv_block(conv_layer_fn, depth, conv_ch, point_wise_type):
    """
    Creates a residual stack of Conv2dNormAct layers.

    If depth == 1, returns only the conv_layer_fn layer (no residuals).

    Args:
        conv_layer_fn (nn.Module): The first Conv2dNormAct layer (already instantiated).
        depth (int): Number of total layers (>= 1).
        conv_ch (int): Channels used for all additional layers.
        point_wise_type: Type of pointwise convolution.

    Returns:
        nn.Module: A single layer or a residual block of layers.
    """
    if depth == 1:
        return conv_layer_fn  # No wrapping needed
    return ResidualConvBlock(conv_layer_fn, depth, conv_ch, point_wise_type)


class DPRNNBlock(nn.Module):
    """
    A single dual-path RNN block. Assumes input channels == hidden_dim.
    Input / output shape: (B, hidden_dim, T, F)
    """
    def __init__(
        self,
        hidden_dim: int,
        num_gru_layers: int = 1,
        stateful: bool = True,
    ):
        super(DPRNNBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        self.stateful = stateful

        # Intra-chunk (feature) RNN: bidirectional
        self.intra_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_intra = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln_intra = nn.LayerNorm(hidden_dim)

        # Inter-chunk (time) RNN: unidirectional
        self.inter_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
        )
        self.inter_states: Optional[torch.Tensor] = None
        self.fc_inter = nn.Linear(hidden_dim, hidden_dim)
        self.ln_inter = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through one DPRNN block.
        Args:
            inputs: Tensor of shape (B, hidden_dim, T, F)
        Returns:
            Tensor of same shape
        """
        B, C, T, F = inputs.shape
        assert C == self.hidden_dim, f"Channel dim must equal hidden_dim ({self.hidden_dim}), got {C}"

        # Intra-chunk (feature) RNN
        x_intra = einops.rearrange(inputs, 'b c t f -> (b t) f c')  # -> (B*T, F, C)
        x_intra, _ = self._execute_rnn(
            x_intra, self.intra_gru, None
        )
        x_intra = self.ln_intra(self.fc_intra(x_intra))             # -> (B*T, F, hidden_dim)
        x_intra = einops.rearrange(x_intra, '(b t) f c -> b c t f',
                                   b=B, t=T, f=F)                 # -> (B, C, T, F)
        x = inputs + x_intra  # residual

        # Inter-chunk (time) RNN
        x_inter = einops.rearrange(x, 'b c t f -> (b f) t c')       # -> (B*F, T, C)
        x_inter, self.inter_states = self._execute_rnn(
            x_inter, self.inter_gru, self.inter_states
        )
        x_inter = self.ln_inter(self.fc_inter(x_inter))              # -> (B*F, T, hidden_dim)
        x_inter = einops.rearrange(x_inter, '(b f) t c -> b c t f',
                                   b=B, t=T, f=F)                 # -> (B, C, T, F)

        return x + x_inter  # residual

    def _execute_rnn(
        self,
        x: torch.Tensor,
        rnn_layer: nn.GRU,
        rnn_states: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Helper to run a GRU layer, with optional statefulness.
        """
        if self.stateful and self.training and rnn_states is not None:
            # If batch size changed, randomly sample existing states
            if x.size(0) != rnn_states.size(1):
                idx = torch.randint(0, rnn_states.size(1), (x.size(0),), device=x.device)
                rnn_states = rnn_states[:, idx]

        if self.stateful and self.training:
            output, next_states = rnn_layer(x, rnn_states)
            next_states = next_states.detach()
        else:
            output, _ = rnn_layer(x)
            next_states = None

        return output, next_states


class DPRNN(nn.Module):
    """
    Stacks multiple DPRNNBlock modules, with input/output channel projection.
    Input shape:  (B, ch_in,  T, F)
    Output shape: (B, ch_out, T, F)
    """
    def __init__(
        self,
        ch_in: int,
        hidden_dim: int,
        ch_out: int,
        num_gru_layers: int = 1,
        num_blocks: int = 6,
        stateful: bool = True,
    ):
        super(DPRNN, self).__init__()
        # project input channels -> hidden_dim
        if ch_in == hidden_dim:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Conv2d(ch_in, hidden_dim, kernel_size=1)

        # stack of DPRNN blocks (each expects hidden_dim channels)
        self.blocks = nn.ModuleList([
            DPRNNBlock(
                hidden_dim=hidden_dim,
                num_gru_layers=num_gru_layers,
                stateful=stateful,
            )
            for _ in range(num_blocks)
        ])

        # project hidden_dim -> output channels
        if hidden_dim == ch_out:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Conv2d(hidden_dim, ch_out, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Tensor of shape (B, ch_in, T, F)
        Returns:
            Tensor of shape (B, ch_out, T, F)
        """
        x = self.input_proj(inputs)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        return x


def init_kernels(win_len, win_inc, fft_len, win_func=None, invers=False):
    if win_func is None:
        window = np.ones(win_len)
    elif isinstance(win_func, str):
        window = get_window(win_func, win_len, fftbins=True) ** 0.5
    else:
        assert isinstance(win_func, np.ndarray) and win_func.ndim == 1
        assert win_func.size == win_len, "win_func must be in the same size as win_len"
        window = win_func ** 0.5

    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T

    if invers:
        kernel = np.linalg.pinv(kernel).T

    kernel = kernel * window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(
        window[None, :, None].astype(np.float32)
    )


class ConvSTFT(nn.Module):
    """
    Taken from https://github.com/wangtianrui/DCCRN/blob/4f961fcb3e431e3d2d4393b40532fe982692b45c/utils/conv_stft.py
    """
    def __init__(
        self,
        win_len,
        win_inc,
        fft_len=None,
        win_func=None,
        return_complex=True,
        fix=True,
        center=True
    ):
        super(ConvSTFT, self).__init__()

        if fft_len is None:
            self.fft_len = int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len

        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_func)
        self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.return_complex = return_complex
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len
        self.center = center

    def forward(self, inputs):
        """
        inputs: B N
        outputs: B F T (complex) | B F T 2 (real)
        """

        if self.center:
            padding = (self.dim // 2, self.dim // 2)
            inputs = torch.nn.functional.pad(inputs, (padding[0], padding[1]), mode='constant', value=0)

        if inputs.dim() == 2:
            inputs = torch.unsqueeze(inputs, 1)

        outputs = F.conv1d(inputs, self.weight, stride=self.stride)

        dim = self.dim // 2 + 1
        real = outputs[:, :dim, :]
        imag = outputs[:, dim:, :]

        if self.return_complex:
            return real + 1j * imag
        else:
            return torch.stack([real, imag], dim=-1)


class ConviSTFT(nn.Module):

    def __init__(self,
                 win_len,
                 win_inc,
                 fft_len=None,
                 win_func='hamming',
                 fix=True
                 ):
        super(ConviSTFT, self).__init__()
        if fft_len == None:
            self.fft_len = int(2 ** np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_func, invers=True)
        # self.weight = nn.Parameter(kernel, requires_grad=(not fix))
        self.register_buffer('weight', kernel)
        self.win_func = win_func
        self.win_len = win_len
        self.stride = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:, None, :])

    def forward(self, inputs):
        """
        inputs : [B, F, T] (complex) or [B, F, T, 2] (real)
        outputs: [B, N]
        """

        if torch.is_complex(inputs):
            inputs = as_real(inputs)   # B, F, T, 2
        real, imag = inputs[..., 0], inputs[..., 1]
        inputs = torch.cat([real, imag], dim=1)  # B, 2F, T
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride)

        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1, 1, inputs.size(-1)) ** 2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs / (coff + 1e-8)
        # outputs = torch.where(coff == 0, outputs, outputs/coff)
        outputs = outputs[..., self.win_len - self.stride:-(self.win_len - self.stride)]
        outputs = outputs[:, 0]     # B, N

        return outputs


class Stft(nn.Module):
    def __init__(self, n_fft: int, win_len: Optional[int] = None,
                 hop: Optional[int] = None, window: Optional[Tensor] = None, normalized: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.win_len = win_len or n_fft
        self.hop = hop or self.win_len // 4
        self.normalized = normalized
        if window is not None:
            assert window.shape[0] == win_len
        else:
            window = torch.hann_window(self.win_len)
        self.w: torch.Tensor
        self.register_buffer("w", window)

    def forward(self, input: Tensor):
        # input: float32 [B, T]
        # output: complex64 [B, F, T']
        out = torch.stft(
            input,
            n_fft=self.n_fft,
            win_length=self.win_len,
            hop_length=self.hop,
            window=self.w,
            normalized=self.normalized,
            return_complex=True,
            center=True
        )
        return out


class Istft(nn.Module):
    def __init__(self, n_fft_inv: int, win_len_inv: Optional[int] = None,
                 hop_inv: Optional[int] = None, window_inv: Optional[Tensor] = None, normalized: bool = True):
        super().__init__()
        # Synthesis back to time domain
        self.n_fft_inv = n_fft_inv
        self.win_len_inv = win_len_inv or n_fft_inv
        self.hop_inv = hop_inv or self.win_len_inv // 4
        self.normalized = normalized

        if window_inv is not None:
            assert window_inv.shape[0] == win_len_inv
        else:
            window_inv = torch.hann_window(self.win_len_inv)
        self.w_inv: torch.Tensor
        self.register_buffer("w_inv", window_inv)

    def forward(self, input: Tensor):
        # input: float32 [B, T, F, (2)] | complex64 [B, T, F]
        # output: float32 [B, T']
        input = as_complex(input)
        out = torch.istft(
            input,
            n_fft=self.n_fft_inv,
            win_length=self.win_len_inv,
            hop_length=self.hop_inv,
            window=self.w_inv,
            normalized=self.normalized,
            center=True,
        )
        return out


class Mask(nn.Module):
    def __init__(self, erb_inv_fb: Tensor, post_filter: bool = False, eps: float = 1e-12):
        super().__init__()
        self.erb_inv_fb: Tensor
        self.register_buffer("erb_inv_fb", erb_inv_fb)
        self.clamp_tensor = torch.__version__ > "1.9.0" or torch.__version__ == "1.9.0"
        self.post_filter = post_filter
        self.eps = eps

    def pf(self, mask: Tensor, beta: float = 0.02) -> Tensor:
        """Post-Filter proposed by Valin et al. [1].

        Args:
            mask (Tensor): Real valued mask, typically of shape [B, C, T, F].
            beta: Global gain factor.
        Refs:
            [1]: Valin et al.: A Perceptually-Motivated Approach for Low-Complexity, Real-Time Enhancement of Fullband Speech.
        """
        mask_sin = mask * torch.sin(np.pi * mask / 2)
        mask_pf = (1 + beta) * mask / (1 + beta * mask.div(mask_sin.clamp_min(self.eps)).pow(2))
        return mask_pf

    def forward(self, spec: Tensor, mask: Tensor, atten_lim: Optional[Tensor] = None) -> Tensor:
        # spec (real) [B, 1, T, F, 2], F: freq_bins
        # mask (real): [B, 1, T, Fe], Fe: erb_bins
        # atten_lim: [B]
        if not self.training and self.post_filter:
            mask = self.pf(mask)
        if atten_lim is not None:
            # dB to amplitude
            atten_lim = 10 ** (-atten_lim / 20)
            # Greater equal (__ge__) not implemented for TorchVersion.
            if self.clamp_tensor:
                # Supported by torch >= 1.9
                mask = mask.clamp(min=atten_lim.view(-1, 1, 1, 1))
            else:
                m_out = []
                for i in range(atten_lim.shape[0]):
                    m_out.append(mask[i].clamp_min(atten_lim[i].item()))
                mask = torch.stack(m_out, dim=0)
        mask = mask.matmul(self.erb_inv_fb)  # [B, 1, T, F]
        if not spec.is_complex():
            mask = mask.unsqueeze(4)
        return spec * mask


class ErbNorm(nn.Module):
    def __init__(self, alpha: float, eps: float = 1e-12, stateful: bool = False,
                 dynamic_var: bool = False):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.init_vals = [-60., -90.]
        self.stateful = stateful
        self.dynamic_var = dynamic_var
        self.mu = None
        self.var = None

    def forward(self, x: Tensor) -> Tensor:
        # x.shape: float32 [B, T, F]
        assert x.ndim == 3, f"input must have 3 dimensions: [B, T, F], {x.ndim} where found."
        B, T, num_feat = x.shape

        if self.mu is None or not self.training or not self.stateful:
            step = (self.init_vals[1] - self.init_vals[0]) / (num_feat - 1)
            mu = self.init_vals[0] + torch.arange(num_feat, device=x.device) * step
            mu = mu[None, :].expand(B, num_feat)
            var = torch.zeros_like(mu) + 40**2  # the init is like the DFN3 default value
        else:
            mu = self.mu
            var = self.var

        x_norm = []
        for t in range(x.shape[1]):
            mu = self.alpha * mu + (1 - self.alpha) * x[:, t]
            if self.dynamic_var:
                var = self.alpha * var + (1 - self.alpha) * ((x[:, t] - mu) ** 2)
            x_norm.append((x[:, t] - mu) / (var.sqrt() + self.eps))
        x_norm = torch.stack(x_norm, dim=1)

        if self.mu is None or not self.training or not self.stateful:
            self.mu = None
            self.var = None
        else:
            self.mu = mu
            self.var = var
        return x_norm


class SpecNorm(nn.Module):
    def __init__(self, alpha: float, eps: float = 1e-12, stateful: bool = False):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.stateful = stateful
        self.init_vals = [0.001, 0.0001]
        self.s = None

    def forward(self, x: Tensor) -> Tensor:
        # x.shape: float32 [B, T, F, 2]
        assert not torch.is_complex(x), f"input must be a float32, not complex. use 'as_real()'"
        assert x.ndim == 4, f"input must have 4 dimensions: [B, T, F, 2], {x.ndim} where found."

        if self.s is None or not self.training or not self.stateful:
            B, T, num_feat, _ = x.shape
            step = (self.init_vals[1] - self.init_vals[0]) / (num_feat - 1)
            s = self.init_vals[0] + torch.arange(num_feat, device=x.device) * step
            s = s[None, :].expand(B, num_feat)
        else:
            s = self.s

        x_abs = as_complex(x).abs()
        x_r_norm = []
        x_i_norm = []
        # s = torch.mean(x_abs, dim=1)
        for t in range(x_abs.shape[1]):
            s = self.alpha * s + (1 - self.alpha) * x_abs[:, t]
            x_r_norm.append(x[:, t, :, 0] / (s + self.eps).sqrt())
            x_i_norm.append(x[:, t, :, 1] / (s + self.eps).sqrt())
        x_r_norm = torch.stack(x_r_norm, dim=1)
        x_i_norm = torch.stack(x_i_norm, dim=1)
        x_norm = torch.stack([x_r_norm, x_i_norm], dim=-1)

        if self.s is None or not self.training or not self.stateful:
            self.s = None
        else:
            self.s = s
        return x_norm


class MagNorm48(nn.Module):
    def __init__(self, alpha: float, eps: float = 1e-12, stateful: bool = False,
                 dynamic_var: bool = False):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.init_vals = InitMagNorm()
        self.stateful = stateful
        self.dynamic_var = dynamic_var
        self.mu = None
        self.var = None

    def forward(self, x: Tensor) -> Tensor:
        # x.shape: float32 [B, T, F]
        assert x.ndim == 3, f"input must have 3 dimensions: [B, T, F], {x.ndim} where found."
        B, T, num_feat = x.shape

        if self.mu is None or not self.training or not self.stateful:
            if num_feat == 481:
                mu = self.init_vals.get_ampirical_mu_0(num_feat)
            else:
                mu = self.init_vals.get_heiuristic_mu_0(num_feat)
            mu = mu[None, :].expand(B, num_feat)
            var = torch.zeros_like(mu) + 40**2  # the init is like the DFN3 default value
        else:
            mu = self.mu
            var = self.var

        x_norm = []
        for t in range(x.shape[1]):
            mu = self.alpha * mu + (1 - self.alpha) * x[:, t]
            if self.dynamic_var:
                var = self.alpha * var + (1 - self.alpha) * ((x[:, t] - mu) ** 2)
            x_norm.append((x[:, t] - mu) / (var.sqrt() + self.eps))
        x_norm = torch.stack(x_norm, dim=1)

        if self.mu is None or not self.training or not self.stateful:
            self.mu = None
            self.var = None
        else:
            self.mu = mu
            self.var = var
        return x_norm


class SpecNorm48(nn.Module):
    def __init__(self, alpha: float, eps: float = 1e-12, stateful: bool = False):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.stateful = stateful
        self.init_vals = InitSpecNorm()
        self.s = None

    def forward(self, x: Tensor) -> Tensor:
        # x.shape: float32 [B, T, F, 2]
        assert not torch.is_complex(x), f"input must be a float32, not complex. use 'as_real()'"
        assert x.ndim == 4, f"input must have 4 dimensions: [B, T, F, 2], {x.ndim} where found."

        if self.s is None or not self.training or not self.stateful:
            B, T, num_feat, _ = x.shape
            if num_feat == 96:
                s = self.init_vals.get_ampirical_s_0(num_feat)
            else:
                s = self.init_vals.get_heiuristic_s_0(num_feat)
            s = s[None, :].expand(B, num_feat)
        else:
            s = self.s

        x_abs = as_complex(x).abs()
        x_r_norm = []
        x_i_norm = []
        # s = torch.mean(x_abs, dim=1)
        for t in range(x_abs.shape[1]):
            s = self.alpha * s + (1 - self.alpha) * x_abs[:, t]
            x_r_norm.append(x[:, t, :, 0] / (s + self.eps).sqrt())
            x_i_norm.append(x[:, t, :, 1] / (s + self.eps).sqrt())
        x_r_norm = torch.stack(x_r_norm, dim=1)
        x_i_norm = torch.stack(x_i_norm, dim=1)
        x_norm = torch.stack([x_r_norm, x_i_norm], dim=-1)

        if self.s is None or not self.training or not self.stateful:
            self.s = None
        else:
            self.s = s
        return x_norm



class Conv2DPointWiseAsLinear(nn.Module):
    def __init__(self, input_channel, output_channel, bias=True):
        super().__init__()
        self.in_channel = input_channel
        self.output_channel = output_channel
        self.cnn_fc = nn.Linear(input_channel, output_channel, bias=bias)
        self.reset_parameters()

    def forward(self, x):
        # input shape should be [B, C, T, F]
        B, C, T, F = x.shape
        x = einops.rearrange(x, 'b c t f -> (b t f) c')
        x = self.cnn_fc(x)
        x = einops.rearrange(x, '(b t f) c -> b c t f', b=B, t=T, f=F)
        return x

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        torch.nn.init.kaiming_uniform_(self.cnn_fc.weight, a=math.sqrt(5))
        if self.cnn_fc.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.cnn_fc.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.cnn_fc.bias, -bound, bound)


class Conv2dNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Iterable[int]],
        fstride: int = 1,
        dilation: int = 1,
        fpad: bool = True,
        bias: bool = True,
        separable: bool = False,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        point_wise_type: str = 'cnn',
    ):
        """Causal Conv2d by delaying the signal for any lookahead.

        Expected input format: [B, C, T, F]
        """
        lookahead = 0  # This needs to be handled on the input feature side
        # Padding on time axis
        kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        if fpad:
            fpad_ = kernel_size[1] // 2 + dilation - 1
        else:
            fpad_ = 0
        pad = (0, 0, kernel_size[0] - 1 - lookahead, lookahead)
        layers = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        if max(kernel_size) == 1:
            separable = False
        if groups > 1 and not (in_ch == out_ch == groups):
            layers.append(
                GroupedConv2D(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    padding=(0, fpad_),
                    stride=(1, fstride),  # Stride over time is always 1
                    dilation=(1, dilation),  # Same for dilation
                    groups=groups,
                    bias=bias,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    padding=(0, fpad_),
                    stride=(1, fstride),  # Stride over time is always 1
                    dilation=(1, dilation),  # Same for dilation
                    groups=groups,
                    bias=bias,
                )
            )
        if separable:
            if point_wise_type == 'cnn':
                layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
            else:
                layers.append(Conv2DPointWiseAsLinear(out_ch, out_ch, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class ConvTranspose2dNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Tuple[int, int]],
        fstride: int = 1,
        dilation: int = 1,
        fpad: bool = True,
        bias: bool = True,
        separable: bool = False,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        point_wise_type: str = 'cnn',
    ):
        """Causal ConvTranspose2d.

        Expected input format: [B, C, T, F]
        """
        # Padding on time axis, with lookahead = 0
        lookahead = 0  # This needs to be handled on the input feature side
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        if fpad:
            fpad_ = kernel_size[1] // 2
        else:
            fpad_ = 0
        pad = (0, 0, kernel_size[0] - 1 - lookahead, lookahead)
        layers = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        layers.append(
            nn.ConvTranspose2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                padding=(kernel_size[0] - 1, fpad_ + dilation - 1),
                output_padding=(0, fpad_),
                stride=(1, fstride),  # Stride over time is always 1
                dilation=(1, dilation),
                groups=groups,
                bias=bias,
            )
        )
        if separable:
            if point_wise_type == 'cnn':
                layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
            else:
                layers.append(Conv2DPointWiseAsLinear(out_ch, out_ch, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class SubPixelConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, fstride=2, padding=(0, 0), dilation=(1, 1), groups=1, bias=True):
        super(SubPixelConv2D, self).__init__()
        assert fstride > 1, "sub-pixel module should expand the f-axis, thus fstride>1"
        self.fstride = fstride
        self.out_channels = out_channels
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
                padding=padding,
                dilation=dilation,
                groups=groups
            ) for _ in range(fstride)
        ])

    def forward(self, inputs):
        out = torch.cat([conv(inputs) for conv in self.convs], dim=1)     # B, S*C, T, F
        out = einops.rearrange(out, 'b (s c) t f -> b c t (f s)', s=self.fstride, c=self.out_channels)
        return out


class SubPixelConv2dNormAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Tuple[int, int]],
        fstride: int = 1,
        dilation: int = 1,
        fpad: bool = True,
        bias: bool = True,
        separable: bool = False,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        point_wise_type: str = 'cnn',
    ):
        """Causal ConvTranspose2d.

        Expected input format: [B, C, T, F]
        """
        # Padding on time axis, with lookahead = 0
        lookahead = 0  # This needs to be handled on the input feature side
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        if fpad:
            fpad_ = kernel_size[1] // 2
        else:
            fpad_ = 0
        pad = (0, 0, kernel_size[0] - 1 - lookahead, lookahead)
        layers = []
        if any(x > 0 for x in pad):
            layers.append(nn.ConstantPad2d(pad, 0.0))
        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        layers.append(
            SubPixelConv2D(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                padding=(0, fpad_ + dilation - 1),
                fstride=fstride,  # Stride over time is always 1
                dilation=(1, dilation),
                groups=groups,
                bias=bias,
            )
        )
        if separable:
            if point_wise_type == 'cnn':
                layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False))
            else:
                layers.append(Conv2DPointWiseAsLinear(out_ch, out_ch, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_ch))
        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)


class GroupedLinearEinsum(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, groups: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.groups = groups
        assert input_size % groups == 0, f"Input size {input_size} not divisible by {groups}"
        assert hidden_size % groups == 0, f"Hidden size {hidden_size} not divisible by {groups}"
        self.ws = input_size // groups
        self.register_parameter(
            "weight",
            nn.Parameter(
                torch.zeros(groups, input_size // groups, hidden_size // groups), requires_grad=True
            ),
        )
        # Register bias parameter
        self.register_parameter(
            "bias",
            nn.Parameter(
                torch.zeros(hidden_size), requires_grad=True
            ),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore
        # Initialize bias as zeros
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        new_shape = (b, t, self.groups, self.ws)
        x = x.view(new_shape)
        x = torch.einsum("btgi,gih->btgh", x, self.weight)  # [..., G, H/G]
        x = x.flatten(2, 3)  # [B, T, H]
        # Add the bias term
        x = x + self.bias
        return x

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(input_size: {self.input_size}, hidden_size: {self.hidden_size}, groups: {self.groups})"


class GroupedLinear(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, groups: int = 1, shuffle: bool = False):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        self.groups = groups
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        if groups == 1:
            shuffle = False
        self.shuffle = shuffle
        self.layers = nn.ModuleList(
            nn.Linear(self.input_size, self.hidden_size) for _ in range(groups)
        )

    def forward(self, x: Tensor) -> Tensor:
        outputs: List[Tensor] = []
        for i, layer in enumerate(self.layers):
            outputs.append(layer(x[..., i * self.input_size: (i + 1) * self.input_size]))
        output = torch.cat(outputs, dim=-1)
        if self.shuffle:
            orig_shape = output.shape
            output = (
                output.view(-1, self.hidden_size, self.groups).transpose(-1, -2).reshape(orig_shape)
            )
        return output

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(input_size: {self.input_size}, hidden_size: {self.hidden_size}, groups: {self.groups})"


class GroupedConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding='valid', dilation=(1, 1), bias=True, groups=1):
        super(GroupedConv2D, self).__init__()
        assert in_ch % groups == 0, "in_ch must be divisible by groups"
        assert out_ch % groups == 0, "out_ch must be divisible by groups"
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.groups = groups
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_ch // groups,
                out_channels=out_ch // groups,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias
            ) for _ in range(groups)
        ])

    def forward(self, x):
        if self.groups > 1:
            input_splits = torch.chunk(x, self.groups, dim=1)
            output_splits = [conv(split) for conv, split in zip(self.convs, input_splits)]
            return torch.cat(output_splits, dim=1)
        else:
            return self.convs[0](x)


class SqueezedGRU(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
        batch_first: bool = True,
        gru_skip_op: Optional[Callable[..., torch.nn.Module]] = None,
        linear_act_layer: Callable[..., torch.nn.Module] = nn.Identity,
        stateful: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_in = nn.Sequential(
            GroupedLinearEinsum(input_size, hidden_size, linear_groups), linear_act_layer()
        )
        self.stateful = stateful
        self.states = None
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                GroupedLinearEinsum(hidden_size, output_size, linear_groups), linear_act_layer()
            )
        else:
            self.linear_out = nn.Identity()

    def forward(self, input: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        input = self.linear_in(input)
        x, self.states = self._execute_rnn(input, self.gru, self.states)
        if self.gru_skip is not None:
            x = x + self.gru_skip(input)
        x = self.linear_out(x)
        return x, h

    def _execute_rnn(self, x: Tensor, rnn_layer: nn.Module,
                    rnn_states: Optional[Tensor] = None) -> [Tensor, Optional[Tensor]]:

        if self.stateful and self.training and rnn_states is not None:
            if x.shape[0] != rnn_states.shape[1]:
                r = torch.randint(0, rnn_states.shape[1], (x.shape[0],))
                rnn_states = rnn_states[:, r]
        if self.stateful and self.training:
            output, nxt_states = rnn_layer(x, rnn_states)
            nxt_states = nxt_states.detach()
        else:
            output, _ = rnn_layer(x)
            nxt_states = None
        return output, nxt_states


class SqueezedGRU_S(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
        batch_first: bool = True,
        gru_skip_op: Optional[Callable[..., torch.nn.Module]] = None,
        linear_act_layer: Callable[..., torch.nn.Module] = nn.Identity,
        group_linear_layer: Callable[..., torch.nn.Module] = GroupedLinearEinsum,
        stateful: bool = False,
        group_gru: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_in = nn.Sequential(
            group_linear_layer(input_size, hidden_size, linear_groups), linear_act_layer()
        )
        self.stateful = stateful
        self.states = None
        gru_class = nn.GRU if group_gru == 1 else GroupedGRU
        self.gru = gru_class(hidden_size, hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                group_linear_layer(hidden_size, output_size, linear_groups), linear_act_layer()
            )
        else:
            self.linear_out = nn.Identity()

    def forward(self, input: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        x = self.linear_in(input)
        x, self.states = self._execute_rnn(x, self.gru, self.states)
        x = self.linear_out(x)
        if self.gru_skip is not None:
            x = x + self.gru_skip(input)
        return x, h

    def _execute_rnn(self, x: Tensor, rnn_layer: nn.Module,
                    rnn_states: Optional[Tensor] = None) -> [Tensor, Optional[Tensor]]:

        if self.stateful and self.training and rnn_states is not None:
            if x.shape[0] != rnn_states.shape[1]:
                r = torch.randint(0, rnn_states.shape[1], (x.shape[0],))
                rnn_states = rnn_states[:, r]
        if self.stateful and self.training:
            output, nxt_states = rnn_layer(x, rnn_states)
            nxt_states = nxt_states.detach()
        else:
            output, _ = rnn_layer(x)
            nxt_states = None
        return output, nxt_states


class GroupedGRULayer(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]
    out_size: Final[int]
    bidirectional: Final[bool]
    num_directions: Final[int]
    groups: Final[int]
    batch_first: Final[bool]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        groups: int,
        batch_first: bool = True,
        bias: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
    ):
        super().__init__()
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        kwargs = {
            "bias": bias,
            "batch_first": batch_first,
            "dropout": dropout,
            "bidirectional": bidirectional,
        }
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        self.out_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.groups = groups
        self.batch_first = batch_first
        assert (self.hidden_size % groups) == 0, "Hidden size must be divisible by groups"
        self.layers = nn.ModuleList(
            (nn.GRU(self.input_size, self.hidden_size, **kwargs) for _ in range(groups))
        )

    def flatten_parameters(self):
        for layer in self.layers:
            layer.flatten_parameters()

    def get_h0(self, batch_size: int = 1, device: torch.device = torch.device("cpu")):
        return torch.zeros(
            self.groups * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )

    def forward(self, input: Tensor, h0: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        # input shape: [B, T, I] if batch_first else [T, B, I], B: batch_size, I: input_size
        # state shape: [G*D, B, H], where G: groups, D: num_directions, H: hidden_size

        if h0 is None:
            dim0, dim1 = input.shape[:2]
            bs = dim0 if self.batch_first else dim1
            h0 = self.get_h0(bs, device=input.device)
        outputs: List[Tensor] = []
        outstates: List[Tensor] = []
        for i, layer in enumerate(self.layers):
            o, s = layer(
                input[..., i * self.input_size : (i + 1) * self.input_size],
                h0[i * self.num_directions : (i + 1) * self.num_directions].detach(),
            )
            outputs.append(o)
            outstates.append(s)
        output = torch.cat(outputs, dim=-1)
        h = torch.cat(outstates, dim=0)
        return output, h


class GroupedGRU(nn.Module):
    groups: Final[int]
    num_layers: Final[int]
    batch_first: Final[bool]
    hidden_size: Final[int]
    bidirectional: Final[bool]
    num_directions: Final[int]
    shuffle: Final[bool]
    add_outputs: Final[bool]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        groups: int = 4,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
        shuffle: bool = True,
        add_outputs: bool = False,
    ):
        super().__init__()
        kwargs = {
            "groups": groups,
            "bias": bias,
            "batch_first": batch_first,
            "dropout": dropout,
            "bidirectional": bidirectional,
        }
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        assert num_layers > 0
        self.input_size = input_size
        self.groups = groups
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.hidden_size = hidden_size // groups
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        if groups == 1:
            shuffle = False  # Fully connected, no need to shuffle
        self.shuffle = shuffle
        self.add_outputs = add_outputs
        self.grus: List[GroupedGRULayer] = nn.ModuleList()  # type: ignore
        self.grus.append(GroupedGRULayer(input_size, hidden_size, **kwargs))
        for _ in range(1, num_layers):
            self.grus.append(GroupedGRULayer(hidden_size, hidden_size, **kwargs))
        self.flatten_parameters()

    def flatten_parameters(self):
        for gru in self.grus:
            gru.flatten_parameters()

    def get_h0(self, batch_size: int, device: torch.device = torch.device("cpu")) -> Tensor:
        return torch.zeros(
            (self.num_layers * self.groups * self.num_directions, batch_size, self.hidden_size),
            device=device,
        )

    def forward(self, input: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        dim0, dim1, _ = input.shape
        b = dim0 if self.batch_first else dim1
        if state is None:
            state = self.get_h0(b, input.device)
        output = torch.zeros(
            dim0, dim1, self.hidden_size * self.num_directions * self.groups, device=input.device
        )
        outstates = []
        h = self.groups * self.num_directions
        for i, gru in enumerate(self.grus):
            input, s = gru(input, state[i * h : (i + 1) * h])
            outstates.append(s)
            if self.shuffle and i < self.num_layers - 1:
                input = (
                    input.view(dim0, dim1, -1, self.groups).transpose(2, 3).reshape(dim0, dim1, -1)
                )
            if self.add_outputs:
                output += input
            else:
                output = input
        outstate = torch.cat(outstates, dim=0)
        return output, outstate


