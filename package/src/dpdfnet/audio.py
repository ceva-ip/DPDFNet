from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


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


def pcm16_safe(audio: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def vorbis_window(window_len: int) -> np.ndarray:
    window_size_h = window_len / 2
    indices = np.arange(window_len)
    s = np.sin(0.5 * np.pi * (indices + 0.5) / window_size_h)
    return np.sin(0.5 * np.pi * s * s).astype(np.float32)


def get_wnorm(window_len: int, frame_size: int) -> float:
    return 1.0 / (window_len**2 / (2 * frame_size))


@dataclass(frozen=True)
class StftConfig:
    win_len: int
    hop_size: int
    window: np.ndarray
    wnorm: float


def make_stft_config(win_len: int) -> StftConfig:
    hop_size = win_len // 2
    window = vorbis_window(win_len)
    wnorm = get_wnorm(win_len, hop_size)
    return StftConfig(win_len=win_len, hop_size=hop_size, window=window, wnorm=wnorm)


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
    spec = (spec.T * cfg.wnorm).astype(np.complex64, copy=False)
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

    waveform_e = waveform_e / cfg.wnorm
    return np.concatenate(
        [waveform_e[cfg.win_len * 2 :], np.zeros(cfg.win_len * 2, dtype=np.float32)],
        axis=0,
    )
