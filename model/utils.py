import numpy as np
import torch
from torch.autograd import Function
from torch.nn import functional as F
from torch import Tensor
from typing import Optional, Tuple


def get_activation(activation: str):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif activation == 'elu':
        return torch.nn.ELU()
    elif activation == 'sigmoid':
        return torch.nn.Sigmoid()
    elif activation == 'gelu':
        return torch.nn.GELU()
    elif activation == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif activation == 'swish':
        return torch.nn.SiLU()
    elif activation == 'identity':
        return torch.nn.Identity()
    else:
        raise NotImplementedError()


def layer_norm(x):
    x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
    return x


def create_comb_filter_matrix(pitch_min=65, pitch_max=500, nfft=512, sample_rate=16000, max_order=None,
                              output_domain='freq', symmetric=False):
    assert output_domain in ['time', 'freq']
    max_order = max_order if max_order is not None else float('inf')
    cf_waveform = torch.zeros(((int(pitch_max) - int(pitch_min) + 1), nfft+1))
    for i, f0 in enumerate(range(int(pitch_min), int(pitch_max)+1, 1)):
        delay = int(sample_rate / f0)
        order = min(int((nfft//2) / delay), max_order)
        weights = torch.hann_window((order + 1) * 2)[order + 1:]
        weights /= weights.sum()
        for o, w in enumerate(weights, 0):
            cf_waveform[i, (nfft // 2) - o*delay] = w

    if symmetric:
        cf_waveform[:, nfft//2 + 1:] = torch.flip(cf_waveform[:, :nfft//2], (-1, ))
        cf_waveform /= cf_waveform.sum(-1, keepdims=True)

    if output_domain == 'time':
        return cf_waveform
    else:
        cf_waveform = cf_waveform[:, :-1]
        hann_window = torch.hann_window(nfft)
        cf_spec = torch.stft(torch.tensor(cf_waveform), nfft, nfft//4, nfft, hann_window, center=False).squeeze()
        cf_mag = torch.view_as_complex(cf_spec).abs()   # size: [CF, nfft//2 + 1]

        # min-max normalization:
        cf_mag = (cf_mag - cf_mag.amin(dim=1, keepdim=True)) / (cf_mag.amax(dim=1, keepdim=True) - cf_mag.amin(dim=1, keepdim=True))

        return cf_mag


def as_complex(x: Tensor) -> Tensor:
    if torch.is_complex(x):
        return x
    if x.shape[-1] != 2:
        raise ValueError(f"Last dimension need to be of length 2 (re + im), but got {x.shape}")
    if x.stride(-1) != 1:
        x = x.contiguous()
    return torch.view_as_complex(x)


def as_real(x: Tensor) -> Tensor:
    if torch.is_complex(x):
        return torch.view_as_real(x)
    return x


def to_db(x: Tensor) -> Tensor:
    return 10 * torch.log10(x + 1e-10)


def power_law_compression(signal: Tensor, alpha: float) -> Tensor:
    return torch.sign(signal) * torch.pow(torch.abs(signal), alpha)


def power_law_decompression(signal: Tensor, alpha: float) -> Tensor:
    return torch.sign(signal) * torch.pow(torch.abs(signal), 1/alpha)


def get_magnitude(x: Tensor) -> Tensor:
    return as_complex(x).abs()


def get_angle(x: Tensor) -> Tensor:
    return angle.apply(as_complex(x))


class angle(Function):
    """Similar to torch.angle but robustify the gradient for zero magnitude."""

    @staticmethod
    def forward(ctx, x: Tensor):
        ctx.save_for_backward(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad: Tensor):
        (x,) = ctx.saved_tensors
        grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(1e-10)
        return torch.view_as_complex(torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1))


def apply_df(spec: Tensor, coefs: Tensor) -> Tensor:
    """Deep filter implementation using `torch.einsum`. Requires unfolded spectrogram.

    Args:
        spec (complex Tensor): Spectrogram of shape [B, C, T, F, N]
        coefs (complex Tensor): Coefficients of shape [B, C, N, T, F]

    Returns:
        spec (complex Tensor): Spectrogram of shape [B, C, T, F]
    """
    return torch.einsum("...tfn,...ntf->...tf", spec, coefs)


def apply_df_real(spec: Tensor, coefs: Tensor) -> Tensor:
    """Deep filter implementation for real valued input Tensors. Requires unfolded spectrograms.

    Args:
        spec (real-valued Tensor): Spectrogram of shape [B, C, N, T, F, 2].
        coefs (real-valued Tensor): Coefficients of shape [B, C, N, T, F, 2].

    Returns:
        spec (real-valued Tensor): Filtered Spectrogram of shape [B, C, T, F, 2]
    """
    b, c, _, t, f, _ = spec.shape
    out = torch.empty((b, c, t, f, 2), dtype=spec.dtype, device=spec.device)
    # real
    out[..., 0] = (spec[..., 0] * coefs[..., 0]).sum(dim=2)
    out[..., 0] -= (spec[..., 1] * coefs[..., 1]).sum(dim=2)
    # imag
    out[..., 1] = (spec[..., 0] * coefs[..., 1]).sum(dim=2)
    out[..., 1] += (spec[..., 1] * coefs[..., 0]).sum(dim=2)
    return out


def vorbis_window(window_len: int) -> Tensor:
    window_size_h = window_len / 2
    # Create the window array
    window = np.zeros(window_len, dtype=np.float32)
    # Fill the window array with the calculated values
    for i in range(window_len):
        sin = np.sin(0.5 * np.pi * (i + 0.5) / window_size_h)
        window[i] = np.sin(0.5 * np.pi * sin * sin)
    return torch.tensor(window)


def get_wnorm(window_len: int, frame_size: int) -> float:
    # window_len - #samples of the window
    # frame_size - hop size
    return 1.0 / (window_len**2 / (2 * frame_size))


def _local_energy(x: Tensor, ws: int, device: torch.device) -> Tensor:
    if (ws % 2) == 0:
        ws += 1
    ws_half = ws // 2
    x = F.pad(x.pow(2).sum(-1).sum(-1), (ws_half, ws_half, 0, 0))
    w = torch.hann_window(ws, device=device, dtype=x.dtype)
    x = x.unfold(-1, size=ws, step=1) * w
    return torch.sum(x, dim=-1).div(ws)


def local_snr(
    clean: Tensor,
    noise: Tensor,
    window_size: int,
    db: bool = False,
    window_size_ns: Optional[int] = None,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, Tensor]:
    # clean shape: [B, C, T, F]
    clean = as_real(clean).unsqueeze(1)
    noise = as_real(noise).unsqueeze(1)
    assert clean.dim() == 5

    E_speech = _local_energy(clean, window_size, clean.device)
    window_size_ns = window_size if window_size_ns is None else window_size_ns
    E_noise = _local_energy(noise, window_size_ns, clean.device)

    snr = E_speech / E_noise.clamp_min(eps)
    if db:
        snr = snr.clamp_min(eps).log10().mul(10)
    return snr, E_speech, E_noise


class LocalSnrTarget(torch.nn.Module):
    def __init__(
            self,
            fft_size: int,
            hop_size: int,
            sr: int,
            ws: int = 20,
            db: bool = True,
            ws_ns: Optional[int] = None,
            target_snr_range=None,
    ):
        super().__init__()
        self.fft_size = fft_size  # length ms of an fft_window
        self.hop_size = hop_size  # consider hop_size
        self.sr = sr
        self.ws = self.calc_ws(ws)
        self.ws_ns = self.ws * 2 if ws_ns is None else self.calc_ws(ws_ns)
        self.db = db
        self.range = target_snr_range

    def calc_ws(self, ws_ms: int) -> int:
        # Calculates windows size in stft domain given a window size in ms
        ws = ws_ms - self.fft_size / self.sr * 1000  # length ms of an fft_window (in samples!)
        ws = 1 + ws / (self.hop_size / self.sr * 1000)  # consider hop_size (in samples!)
        return max(int(round(ws)), 1)

    def forward(self, clean: Tensor, noise: Tensor, max_bin: Optional[int] = None) -> Tensor:
        # clean: [B, 1, T, F]
        # out: [B, T']
        if max_bin is not None:
            clean = as_complex(clean[..., :max_bin])
            noise = as_complex(noise[..., :max_bin])
        return (
            local_snr(clean, noise, window_size=self.ws, db=self.db, window_size_ns=self.ws_ns)[0]
            .clamp(self.range[0], self.range[1])
            .squeeze(1)
        )


def hz2erb(hz):
    """
    Convert a value in Hertz to ERBs
    Args:
         hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    Returns:
        a value in ERBs. If an array was passed in, an identical sized array is returned.
    """
    return 9.265 * np.log1p(hz / (24.7 * 9.265))


def erb2hz(erb):
    """
    Convert a value in ERBs to Hertz
    Args:
        erb: a value in ERBs. This can also be a numpy array, conversion
             proceeds element-wise.
    Returns:
        a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 24.7 * 9.265 * (np.exp(erb / 9.265) - 1.0)


def erb_filter_banks(n_filters: int = 32,
                     nfft: int = 512,
                     fs: int = 16000,
                     low_freq: int = 0,
                     high_freq: Optional[int] = None,
                     min_nb_freqs: int = 2):
    """
    Compute ERB-filterbanks. The filters are stored in the rows, the columns
    correspond to fft bins.
    Args:
        n_filters (int, default=40): the number of filters in the filterbank.
        nfft (int, default=512): the FFT size.
        fs (int, default=16000): sample rate/ sampling frequency of the signal.
        low_freq (int, default=0): lowest band edge of mel filters.
        high_freq (int, optional): highest band edge of mel filters. (Default samplerate/2)
        min_nb_freqs: (int, default=2): the minimum number of freq-bins per ERB.

    Returns:
        fbank: a numpy array of size n_filters * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    def freq2erb(freq):
        return 9.265 * np.log1p(freq / (24.7 * 9.265))

    def erb2freq(erb):
        return 24.7 * 9.265 * (np.exp(erb / 9.265) - 1)

    # init freqs
    high_freq = high_freq if high_freq else fs // 2

    # run checks
    assert high_freq <= (fs // 2), "high frequency can not be greater than the maximum frequency."
    assert 0 <= low_freq < high_freq, "low frequency must be between 0 to high_freq - 1."

    nyq_freq = fs / 2
    freq_width = fs / nfft
    erb_low = freq2erb(0.0)
    erb_high = freq2erb(nyq_freq)
    step = (erb_high - erb_low) / n_filters
    bins = np.zeros(n_filters + 1, dtype=np.int32)
    for i in range(33):
        bins[i] = int(round(erb2freq(erb_low + i * step) / freq_width))
    bins[-1] = nfft // 2 + 1
    bins = bins

    # compute amps of fbanks
    fbank = np.zeros([n_filters, nfft // 2 + 1])
    freq_over = 0
    for j in range(0, n_filters):
        alpha, beta = bins[j] + freq_over, bins[j + 1]
        if (beta - alpha) < min_nb_freqs:
            freq_over = min_nb_freqs - (beta - alpha)
            beta = min(beta + freq_over, nfft // 2 + 1)
        else:
            freq_over = 0

        # compute fbank bins
        fbank[j, alpha:beta] = 1.0

    assert (fbank.sum(axis=1) > 0).all(), "Some rows in fbank are all zeros, please decrease number of erbs or increase nfft"
    return np.abs(fbank)
