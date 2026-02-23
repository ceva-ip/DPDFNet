import math
import numpy as np
import einops
import torch
import torch.nn as nn
from typing import Optional, Final, Callable, Tuple, Union, Iterable, List
from torch import Tensor

from model.utils import as_complex
from streaming.init_norms import InitMagNorm, InitSpecNorm
from streaming.utils import get_mag


class CyclicBuffer(nn.Module):
    def __init__(self, shape: list, time_steps: int, delay_frames: int = 0, time_dim: int = 2):
        super().__init__()
        if time_steps < 1:
            raise ValueError("time_steps must be >= 1")
        if delay_frames < 0:
            raise ValueError("delay_frames must be >= 0")

        self.time_steps = int(time_steps)
        self.delay_frames = int(delay_frames)
        self.time_dim = int(time_dim)
        self.capacity = self.time_steps + self.delay_frames

        # Work on a copy so we don't mutate caller's list
        shape = list(shape)

        # Normalize negative dims (PyTorch-style)
        if self.time_dim < 0:
            self.time_dim = len(shape) + self.time_dim
        if not (0 <= self.time_dim < len(shape)):
            raise ValueError(f"time_dim={time_dim} out of range for shape with {len(shape)} dims.")

        # Set the time dimension to capacity in original layout
        shape[self.time_dim] = self.capacity

        # Permutation that moves time_dim -> 0 (time-first)
        dims = list(range(len(shape)))
        self.perm = [self.time_dim] + [d for d in dims if d != self.time_dim]
        self.inv_perm = [self.perm.index(d) for d in dims]

        # Allocate metadata for state tensor in time-first layout.
        self.shape_tf = tuple(shape[d] for d in self.perm)  # time is now dim 0
        self._state_size = math.prod(self.shape_tf)

    def state_size(self) -> int:
        return self._state_size

    def initial_state(
            self,
            state: Optional[Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if state is None:
            return torch.zeros(self._state_size, dtype=dtype, device=device)
        state = state.reshape(-1)
        if state.numel() != self._state_size:
            raise ValueError(f"Initial state size mismatch in CyclicBuffer: expected {self._state_size}, got {state.numel()}")
        return state

    def reset(self):
        # No persistent state to reset. Kept for API compatibility.
        return

    def forward(
            self,
            x: torch.Tensor,
            state: Optional[Tensor] = None,
            offset: int = 0
    ) -> Union[Tensor, Tuple[Tensor, Tensor, int]]:
        # Validate dimensionality
        if x.dim() != len(self.inv_perm):
            raise ValueError(f"Expected input with {len(self.inv_perm)} dims, got {x.dim()} dims.")

        # Input must carry a single frame along the original time_dim
        if x.shape[self.time_dim] != 1:
            raise ValueError(
                f"Input must have a single frame along time_dim={self.time_dim}; got {x.shape[self.time_dim]}."
            )

        # Move time_dim -> 0 for both update and slicing (time-first view)
        x_tf = x.permute(self.perm)  # shape: [1, ...]
        if state is None:
            buf = x_tf.new_zeros(self.shape_tf)
        else:
            end = offset + self._state_size
            buf_flat = state[offset:end]
            if buf_flat.numel() != self._state_size:
                raise ValueError(f"Insufficient state size in CyclicBuffer: need {self._state_size}, got {buf_flat.numel()}")
            buf = buf_flat.view(self.shape_tf)

        # -------------------------
        # FIFO update (drop oldest, append newest)
        # -------------------------
        # buf[1:] drops the oldest frame at t=0
        buf_next = torch.cat([buf[1:], x_tf], dim=0)

        # Return the first `time_steps` frames (time-first), then restore original layout
        out_tf = buf_next[:self.time_steps]
        out = out_tf.permute(self.inv_perm)
        if state is None:
            return out
        state_out = buf_next.reshape(-1)
        return out, state_out, offset + self._state_size

class DPRNNBlock(nn.Module):
    """
    A single dual-path RNN block. Assumes input channels == hidden_dim.
    Input / output shape: (B=1, hidden_dim, T=1, F)
    """
    def __init__(
        self,
        num_feat: int,
        hidden_dim: int,
        stateful: bool = True,
    ):
        super(DPRNNBlock, self).__init__()
        self.num_feat = num_feat
        self.hidden_dim = hidden_dim
        self.stateful = stateful

        # Intra-chunk (feature) RNN: bidirectional
        self.intra_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_intra = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln_intra = nn.LayerNorm(hidden_dim)

        # Inter-chunk (time) RNN: unidirectional
        self.inter_gru = GRUCellInternalState(hidden_dim, hidden_dim, batch_size=num_feat)
        self.fc_inter = nn.Linear(hidden_dim, hidden_dim)
        self.ln_inter = nn.LayerNorm(hidden_dim)

    def state_size(self) -> int:
        return self.inter_gru.state_size()

    def initial_state(
            self,
            state: Optional[Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(
                    f"Initial state size mismatch in DPRNNBlock: expected {self.state_size()}, got {state.numel()}"
                )
            return state
        return self.inter_gru.initial_state(device=device, dtype=dtype)

    def forward(
            self,
            inputs: torch.Tensor,
            state: Optional[Tensor] = None,
            offset: int = 0
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tensor, int]]:
        """
        Forward pass through one DPRNN block.
        Args:
            inputs: Tensor of shape (B=1, hidden_dim, T=1, F)
        Returns:
            Tensor of same shape
        """
        B, C, T, F = inputs.shape
        assert C == self.hidden_dim, f"Channel dim must equal hidden_dim ({self.hidden_dim}), got {C}"

        # Intra-chunk (feature) RNN
        x_intra = einops.rearrange(inputs, 'b c t f -> (b t) f c')  # -> (B*T, F, C)
        x_intra, _ = self.intra_gru(x_intra)
        x_intra = self.ln_intra(self.fc_intra(x_intra))             # -> (B*T, F, hidden_dim)
        x_intra = einops.rearrange(x_intra, '(b t) f c -> b c t f',
                                   b=B, t=T, f=F)                 # -> (B=1, C, T=1, F)
        x = inputs + x_intra  # residual

        # Inter-chunk (time) RNN
        x_inter = einops.rearrange(x, 'b c t f -> (b f t) c')       # -> (B*F*T, C)
        if state is None:
            x_inter = self.inter_gru(x_inter)
            inter_state_out = None
        else:
            x_inter, inter_state_out, offset = self.inter_gru(x_inter, state=state, offset=offset)
        x_inter = self.ln_inter(self.fc_inter(x_inter))              # -> (B*F*T, hidden_dim)
        x_inter = einops.rearrange(x_inter, '(b f t) c -> b c t f',
                                   b=B, t=T, f=F)                 # -> (B=1, C, T=1, F)
        y = x + x_inter  # residual
        if state is None:
            return y
        return y, inter_state_out, offset

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
        num_feat: int,
        ch_in: int,
        hidden_dim: int,
        ch_out: int,
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
                num_feat=num_feat,
                hidden_dim=hidden_dim,
                stateful=stateful,
            )
            for _ in range(num_blocks)
        ])

        # project hidden_dim -> output channels
        if hidden_dim == ch_out:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Conv2d(hidden_dim, ch_out, kernel_size=1)

    def state_size(self) -> int:
        return sum(block.state_size() for block in self.blocks)

    def initial_state(
            self,
            state: Optional[Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"Initial state size mismatch in DPRNN: expected {self.state_size()}, got {state.numel()}")
            return state
        chunks = [block.initial_state(device=device, dtype=dtype) for block in self.blocks]
        return torch.cat(chunks, dim=0) if chunks else torch.zeros(0, dtype=dtype, device=device)

    def forward(
            self,
            inputs: torch.Tensor,
            state: Optional[Tensor] = None,
            offset: int = 0
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tensor, int]]:
        """
        Args:
            inputs: Tensor of shape (B, ch_in, T, F)
        Returns:
            Tensor of shape (B, ch_out, T, F)
        """
        x = self.input_proj(inputs)
        state_out_chunks: List[Tensor] = []
        for block in self.blocks:
            if state is None:
                x = block(x)
            else:
                x, block_state_out, offset = block(x, state=state, offset=offset)
                state_out_chunks.append(block_state_out)
        x = self.output_proj(x)
        if state is None:
            return x
        state_out = torch.cat(state_out_chunks, dim=0) if state_out_chunks else state.new_zeros(0)
        return x, state_out, offset


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
        self.spec_buffer = CyclicBuffer(
            # [B, 1, T, F, 2]
            shape=[1, 1, 1, erb_inv_fb.shape[1], 2],
            time_steps=1,
            delay_frames=2,
            time_dim=2
        )

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

    def state_size(self) -> int:
        return self.spec_buffer.state_size()

    def initial_state(
            self,
            state: Optional[Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"Initial state size mismatch in Mask: expected {self.state_size()}, got {state.numel()}")
            return state
        return self.spec_buffer.initial_state(device=device, dtype=dtype)

    def forward(
            self,
            spec: Tensor,
            mask: Tensor,
            atten_lim: Optional[Tensor] = None,
            state: Optional[Tensor] = None,
            offset: int = 0
    ) -> Union[Tensor, Tuple[Tensor, Tensor, int]]:
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
        if state is None:
            return self.spec_buffer(spec) * mask
        buffered_spec, state_out, offset = self.spec_buffer(spec, state=state, offset=offset)
        return buffered_spec * mask, state_out, offset


class ErbNorm(nn.Module):
    def __init__(self, num_feat: int, alpha: float, eps: float = 1e-12, stateful: bool = False,
                 dynamic_var: bool = False):
        super().__init__()
        self.num_feat = num_feat
        self.alpha = alpha
        self.eps = eps
        self.init_vals = [-60., -90.]
        self.stateful = stateful
        self.dynamic_var = dynamic_var
        self.register_buffer("mu0", self._init_state(), persistent=False)

    def _init_state(self):
        step = (self.init_vals[1] - self.init_vals[0]) / (self.num_feat - 1)
        mu = self.init_vals[0] + torch.arange(self.num_feat) * step
        return torch.reshape(mu, [1, 1, self.num_feat])

    def state_size(self) -> int:
        return self.num_feat

    def initial_state(
            self,
            state: Optional[Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"Initial state size mismatch in ErbNorm: expected {self.state_size()}, got {state.numel()}")
            return state
        return self.mu0.reshape(-1).clone()

    def reset(self):
        # No persistent state to reset. Kept for API compatibility.
        return

    def forward(
            self,
            x: Tensor,
            state: Optional[Tensor] = None,
            offset: int = 0
    ) -> Union[Tensor, Tuple[Tensor, Tensor, int]]:
        # x.shape: float32 [B=1, T=1, F_erb]
        assert x.ndim == 3, f"input must have 3 dimensions: [B, T, F], {x.ndim} where found."
        if state is None:
            mu = self.mu0
            mu = self.alpha * mu + (1 - self.alpha) * x
            return (x - mu) / 40.

        end = offset + self.state_size()
        mu_flat = state[offset:end]
        if mu_flat.numel() != self.state_size():
            raise ValueError(f"Insufficient state size in ErbNorm: need {self.state_size()}, got {mu_flat.numel()}")
        mu_prev = mu_flat.view(1, 1, self.num_feat)
        mu_next = self.alpha * mu_prev + (1 - self.alpha) * x
        x_norm = (x - mu_next) / 40.
        state_out = mu_next.reshape(-1)
        return x_norm, state_out, end


class SpecNorm(nn.Module):
    def __init__(self, num_feat: int, alpha: float, eps: float = 1e-12, stateful: bool = False):
        super().__init__()
        self.num_feat = num_feat
        self.alpha = alpha
        self.eps = eps
        self.stateful = stateful
        self.init_vals = [0.001, 0.0001]
        self.register_buffer("s0", self._init_state(), persistent=False)

    def _init_state(self):
        step = (self.init_vals[1] - self.init_vals[0]) / (self.num_feat - 1)
        s = self.init_vals[0] + torch.arange(self.num_feat) * step
        return torch.reshape(s, [1, 1, self.num_feat])

    def state_size(self) -> int:
        return self.num_feat

    def initial_state(
            self,
            state: Optional[Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(f"Initial state size mismatch in SpecNorm: expected {self.state_size()}, got {state.numel()}")
            return state
        return self.s0.reshape(-1).clone()

    def reset(self):
        # No persistent state to reset. Kept for API compatibility.
        return

    def forward(
            self,
            x: Tensor,
            state: Optional[Tensor] = None,
            offset: int = 0
    ) -> Union[Tensor, Tuple[Tensor, Tensor, int]]:
        # x.shape: float32 [B=1, T=1, F_df, 2]
        assert x.ndim == 4, f"input must have 4 dimensions: [B, T, F, 2], {x.ndim} where found."
        x_abs = get_mag(x)
        if state is None:
            s = self.s0
            s = self.alpha * s + (1 - self.alpha) * x_abs
            denom = (s + self.eps).sqrt()
            x_r_norm = x[..., 0] / denom
            x_i_norm = x[..., 1] / denom
            return torch.stack([x_r_norm, x_i_norm], dim=-1)

        end = offset + self.state_size()
        s_flat = state[offset:end]
        if s_flat.numel() != self.state_size():
            raise ValueError(f"Insufficient state size in SpecNorm: need {self.state_size()}, got {s_flat.numel()}")
        s_prev = s_flat.view(1, 1, self.num_feat)
        s_next = self.alpha * s_prev + (1 - self.alpha) * x_abs
        denom = (s_next + self.eps).sqrt()
        x_r_norm = x[..., 0] / denom
        x_i_norm = x[..., 1] / denom
        x_norm = torch.stack([x_r_norm, x_i_norm], dim=-1)
        state_out = s_next.reshape(-1)
        return x_norm, state_out, end


class MagNorm48(nn.Module):
    def __init__(self, num_feat: int, alpha: float, eps: float = 1e-12, stateful: bool = False,
                 dynamic_var: bool = False):
        super().__init__()
        self.num_feat = num_feat
        self.alpha = alpha
        self.eps = eps
        self.stateful = stateful
        self.dynamic_var = dynamic_var
        self.init_vals = InitMagNorm()
        self.register_buffer("mu0", self._init_mu0(), persistent=False)
        self.register_buffer("var0", torch.full((1, 1, self.num_feat), 40 ** 2), persistent=False)

    def _init_mu0(self) -> Tensor:
        if self.num_feat == 481:
            mu = self.init_vals.get_ampirical_mu_0(num_feat=self.num_feat)
        else:
            mu = self.init_vals.get_heiuristic_mu_0(num_feat=self.num_feat)
        return mu.view(1, 1, self.num_feat)

    def state_size(self) -> int:
        if self.dynamic_var:
            return 2 * self.num_feat
        return self.num_feat

    def initial_state(
            self,
            state: Optional[Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(
                    f"Initial state size mismatch in MagNorm48: expected {self.state_size()}, got {state.numel()}"
                )
            return state
        if self.dynamic_var:
            return torch.cat([self.mu0.reshape(-1), self.var0.reshape(-1)], dim=0).clone()
        return self.mu0.reshape(-1).clone()

    def reset(self):
        # No persistent state to reset. Kept for API compatibility.
        return

    def forward(
            self,
            x: Tensor,
            state: Optional[Tensor] = None,
            offset: int = 0
    ) -> Union[Tensor, Tuple[Tensor, Tensor, int]]:
        # x.shape: float32 [B=1, T=1, F]
        assert x.ndim == 3, f"input must have 3 dimensions: [B, T, F], {x.ndim} where found."
        if state is None:
            mu_next = self.alpha * self.mu0 + (1 - self.alpha) * x
            if self.dynamic_var:
                var_next = self.alpha * self.var0 + (1 - self.alpha) * ((x - mu_next) ** 2)
            else:
                var_next = self.var0
            return (x - mu_next) / (var_next.sqrt() + self.eps)

        end = offset + self.num_feat
        mu_flat = state[offset:end]
        if mu_flat.numel() != self.num_feat:
            raise ValueError(f"Insufficient state size in MagNorm48: need {self.num_feat}, got {mu_flat.numel()}")
        mu_prev = mu_flat.view(1, 1, self.num_feat)
        mu_next = self.alpha * mu_prev + (1 - self.alpha) * x

        if self.dynamic_var:
            var_end = end + self.num_feat
            var_flat = state[end:var_end]
            if var_flat.numel() != self.num_feat:
                raise ValueError(
                    f"Insufficient variance state size in MagNorm48: need {self.num_feat}, got {var_flat.numel()}"
                )
            var_prev = var_flat.view(1, 1, self.num_feat)
            var_next = self.alpha * var_prev + (1 - self.alpha) * ((x - mu_next) ** 2)
            next_offset = var_end
            state_out = torch.cat([mu_next.reshape(-1), var_next.reshape(-1)], dim=0)
        else:
            var_next = self.var0
            next_offset = end
            state_out = mu_next.reshape(-1)

        x_norm = (x - mu_next) / (var_next.sqrt() + self.eps)
        return x_norm, state_out, next_offset


class SpecNorm48(nn.Module):
    def __init__(self, num_feat: int, alpha: float, eps: float = 1e-12, stateful: bool = False):
        super().__init__()
        self.num_feat = num_feat
        self.alpha = alpha
        self.eps = eps
        self.stateful = stateful
        self.init_vals = InitSpecNorm()
        self.register_buffer("s0", self._init_state(), persistent=False)

    def _init_state(self) -> Tensor:
        if self.num_feat == 96:
            s = self.init_vals.get_ampirical_s_0(num_feat=self.num_feat)
        else:
            s = self.init_vals.get_heiuristic_s_0(num_feat=self.num_feat)
        return s.view(1, 1, self.num_feat)

    def state_size(self) -> int:
        return self.num_feat

    def initial_state(
            self,
            state: Optional[Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(
                    f"Initial state size mismatch in SpecNorm48: expected {self.state_size()}, got {state.numel()}"
                )
            return state
        return self.s0.reshape(-1).clone()

    def reset(self):
        # No persistent state to reset. Kept for API compatibility.
        return

    def forward(
            self,
            x: Tensor,
            state: Optional[Tensor] = None,
            offset: int = 0
    ) -> Union[Tensor, Tuple[Tensor, Tensor, int]]:
        # x.shape: float32 [B=1, T=1, F, 2]
        assert x.ndim == 4, f"input must have 4 dimensions: [B, T, F, 2], {x.ndim} where found."
        x_abs = get_mag(x)
        if state is None:
            s_next = self.alpha * self.s0 + (1 - self.alpha) * x_abs
            denom = (s_next + self.eps).sqrt()
            x_r_norm = x[..., 0] / denom
            x_i_norm = x[..., 1] / denom
            return torch.stack([x_r_norm, x_i_norm], dim=-1)

        end = offset + self.state_size()
        s_flat = state[offset:end]
        if s_flat.numel() != self.state_size():
            raise ValueError(f"Insufficient state size in SpecNorm48: need {self.state_size()}, got {s_flat.numel()}")
        s_prev = s_flat.view(1, 1, self.num_feat)
        s_next = self.alpha * s_prev + (1 - self.alpha) * x_abs
        denom = (s_next + self.eps).sqrt()
        x_r_norm = x[..., 0] / denom
        x_i_norm = x[..., 1] / denom
        x_norm = torch.stack([x_r_norm, x_i_norm], dim=-1)
        state_out = s_next.reshape(-1)
        return x_norm, state_out, end


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
            # layers.append(nn.ConstantPad2d(pad, 0.0))
            layers.append(nn.Identity())
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
        gru_skip_op: Optional[Callable[..., torch.nn.Module]] = None,
        linear_act_layer: Callable[..., torch.nn.Module] = nn.Identity,
        group_linear_layer: Callable[..., torch.nn.Module] = GroupedLinearEinsum,
        stateful: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_in = nn.Sequential(
            group_linear_layer(input_size, hidden_size, linear_groups), linear_act_layer()
        )
        self.stateful = stateful
        self.gru = nn.ModuleList([GRUCellInternalState(hidden_size, hidden_size) for _ in range(num_layers)])
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                group_linear_layer(hidden_size, output_size, linear_groups), linear_act_layer()
            )
        else:
            self.linear_out = nn.Identity()

    def state_size(self) -> int:
        return sum(gru_cell.state_size() for gru_cell in self.gru)

    def initial_state(
            self,
            state: Optional[Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(
                    f"Initial state size mismatch in SqueezedGRU_S: expected {self.state_size()}, got {state.numel()}"
                )
            return state
        chunks = [gru_cell.initial_state(device=device, dtype=dtype) for gru_cell in self.gru]
        return torch.cat(chunks, dim=0) if chunks else torch.zeros(0, dtype=dtype, device=device)

    def forward(
            self,
            inputs: Tensor,
            state: Optional[Tensor] = None,
            offset: int = 0
    ) -> Union[Tensor, Tuple[Tensor, Tensor, int]]:
        x = self.linear_in(inputs)
        state_out_chunks: List[Tensor] = []
        for gru_cell in self.gru:
            if state is None:
                x = gru_cell(x)
            else:
                x, gru_state_out, offset = gru_cell(x, state=state, offset=offset)
                state_out_chunks.append(gru_state_out)
        x = self.linear_out(x)
        if self.gru_skip is not None:
            x = x + self.gru_skip(inputs)
        if state is None:
            return x
        state_out = torch.cat(state_out_chunks, dim=0) if state_out_chunks else state.new_zeros(0)
        return x, state_out, offset

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


class GRUCellInternalState(nn.Module):
    def __init__(self, input_size: int, units: int, batch_size: int = 1):
        super().__init__()
        self.units = units
        self.batch_size = batch_size
        self.grucell = nn.GRUCell(input_size=input_size, hidden_size=units)

    def state_size(self) -> int:
        return self.batch_size * self.units

    def initial_state(
            self,
            state: Optional[Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
    ) -> Tensor:
        if state is not None:
            state = state.reshape(-1)
            if state.numel() != self.state_size():
                raise ValueError(
                    f"Initial state size mismatch in GRUCellInternalState: expected {self.state_size()}, got {state.numel()}"
                )
            return state
        return torch.zeros(self.state_size(), dtype=dtype, device=device)

    def reset(self):
        # No persistent state to reset. Kept for API compatibility.
        return

    def forward(
            self,
            inputs: Tensor,
            state: Optional[Tensor] = None,
            offset: int = 0
    ) -> Union[Tensor, Tuple[Tensor, Tensor, int]]:
        """
        inputs: (B=1, units)
        """
        if state is None:
            h_prev = inputs.new_zeros(inputs.shape[0], self.units)
            return self.grucell(inputs, h_prev)

        if inputs.shape[0] != self.batch_size:
            raise ValueError(f"GRUCellInternalState expected batch_size={self.batch_size}, got {inputs.shape[0]}")
        end = offset + self.state_size()
        h_flat = state[offset:end]
        if h_flat.numel() != self.state_size():
            raise ValueError(
                f"Insufficient state size in GRUCellInternalState: need {self.state_size()}, got {h_flat.numel()}"
            )
        h_prev = h_flat.view(self.batch_size, self.units)
        y = self.grucell(inputs, h_prev)
        state_out = y.reshape(-1)
        return y, state_out, end
