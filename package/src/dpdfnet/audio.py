from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

ATTN_LIMIT_NOISY_FRAME_OFFSET = 4


def to_mono(audio: np.ndarray) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.ndim != 2:
        raise ValueError(f"Expected mono/stereo audio, got shape {x.shape}")
    return np.mean(x, axis=1, dtype=np.float32)


def ensure_sample_rate(audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if sample_rate == target_sample_rate:
        return np.asarray(audio, dtype=np.float32)
    return librosa.resample(
        np.asarray(audio, dtype=np.float32),
        orig_sr=sample_rate,
        target_sr=target_sample_rate,
    ).astype(np.float32, copy=False)


def fit_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.shape[0] == target_len:
        return x
    if x.shape[0] > target_len:
        return x[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[: x.shape[0]] = x
    return out


def _validate_attn_limit_db(attn_limit_db: float | None) -> float | None:
    if attn_limit_db is None:
        return None
    value = float(attn_limit_db)
    if np.isnan(value) or value < 0.0:
        raise ValueError("attn_limit_db must be non-negative, infinity, or None.")
    return value


def apply_attn_limit(
    spec_noisy: np.ndarray,
    spec_enh: np.ndarray,
    attn_limit_db: float | None,
) -> np.ndarray:
    value = _validate_attn_limit_db(attn_limit_db)
    enhanced = np.asarray(spec_enh, dtype=np.float32)
    if value is None:
        return enhanced

    noisy = np.asarray(spec_noisy, dtype=np.float32)
    if noisy.shape != enhanced.shape:
        raise ValueError(
            "spec_noisy and spec_enh must have matching shapes, "
            f"got {noisy.shape} and {enhanced.shape}."
        )

    # The offline ISTFT path advances the output by ~4 hops, so align the
    # noisy reference to that frame index before attenuation-limit blending.
    aligned_noisy = np.zeros_like(noisy, dtype=np.float32)
    if noisy.shape[1] > ATTN_LIMIT_NOISY_FRAME_OFFSET:
        aligned_noisy[:, ATTN_LIMIT_NOISY_FRAME_OFFSET:, :, :] = noisy[
            :, :-ATTN_LIMIT_NOISY_FRAME_OFFSET, :, :
        ]

    alpha = float(10.0 ** (-value / 20.0))
    return np.ascontiguousarray(alpha * aligned_noisy + (1.0 - alpha) * enhanced, dtype=np.float32)


def pcm16_safe(audio: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def vorbis_window(window_len: int) -> np.ndarray:
    window_size_h = window_len / 2
    indices = np.arange(window_len)
    s = np.sin(0.5 * np.pi * (indices + 0.5) / window_size_h)
    return np.sin(0.5 * np.pi * s * s).astype(np.float32)


@dataclass(frozen=True)
class StftConfig:
    win_len: int
    hop_size: int
    window: np.ndarray


def make_stft_config(win_len: int) -> StftConfig:
    hop_size = win_len // 2
    window = vorbis_window(win_len)
    return StftConfig(win_len=win_len, hop_size=hop_size, window=window)


def preprocess_waveform(waveform: np.ndarray, cfg: StftConfig) -> np.ndarray:
    x = np.asarray(waveform, dtype=np.float32).reshape(-1)
    spec = librosa.stft(
        y=x,
        n_fft=cfg.win_len,
        hop_length=cfg.hop_size,
        win_length=cfg.win_len,
        window=cfg.window,
        center=True,
        pad_mode="reflect",
    )
    spec = spec.T.astype(np.complex64, copy=False)
    spec_ri = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32, copy=False)
    return spec_ri[None, ...]


def postprocess_spec(spec_e: np.ndarray, cfg: StftConfig) -> np.ndarray:
    spec_c = np.asarray(spec_e[0], dtype=np.float32)
    spec = (spec_c[..., 0] + 1j * spec_c[..., 1]).T.astype(np.complex64, copy=False)

    waveform_e = librosa.istft(
        spec,
        hop_length=cfg.hop_size,
        win_length=cfg.win_len,
        window=cfg.window,
        center=True,
        length=None,
    ).astype(np.float32, copy=False)

    return np.concatenate(
        [waveform_e[cfg.win_len * 2 :], np.zeros(cfg.win_len * 2, dtype=np.float32)],
        axis=0,
    )
