import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf

from banner import print_banner

TFLITE_DIR = Path('./model_zoo/tflite')
Interpreter = tf.lite.Interpreter
ATTN_LIMIT_NOISY_FRAME_OFFSET = 4

# -----------------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------------
# 16 kHz models: WIN_LEN=320  (20 ms)
# 48 kHz models: WIN_LEN=960  (20 ms)

MODEL_CONFIG = {
    # 16 kHz models
    "baseline":  {"sr": 16000, "win_len": 320},
    "dpdfnet2":  {"sr": 16000, "win_len": 320},
    "dpdfnet4":  {"sr": 16000, "win_len": 320},
    "dpdfnet8":  {"sr": 16000, "win_len": 320},

    # 48 kHz models - TBD
    "dpdfnet2_48khz_hr": {"sr": 48000, "win_len": 960},
    "dpdfnet8_48khz_hr": {"sr": 48000, "win_len": 960},
}


def vorbis_window(window_len: int) -> np.ndarray:
    window_size_h = window_len / 2
    indices = np.arange(window_len)
    sin = np.sin(0.5 * np.pi * (indices + 0.5) / window_size_h)
    window = np.sin(0.5 * np.pi * sin * sin)
    return window.astype(np.float32)


def get_wnorm(window_len: int, frame_size: int) -> float:
    return 1.0 / (window_len ** 2 / (2 * frame_size))


@dataclass(frozen=True)
class STFTConfig:
    sr: int
    win_len: int
    hop_size: int
    win: np.ndarray
    wnorm: float


def make_stft_config(sr: int, win_len: int) -> STFTConfig:
    hop_size = win_len // 2
    win = vorbis_window(win_len)
    wnorm = get_wnorm(win_len, hop_size)
    return STFTConfig(sr=sr, win_len=win_len, hop_size=hop_size, win=win, wnorm=wnorm)


# -----------------------------------------------------------------------------
# Pre/Post processing
# -----------------------------------------------------------------------------

def preprocessing(waveform: np.ndarray, cfg: STFTConfig) -> np.ndarray:
    """
    waveform: 1D float32 numpy array at cfg.sr, mono, range ~[-1,1]
    Returns complex STFT as real/imag split: [B=1, T, F, 2] float32
    """
    spec = librosa.stft(
        y=waveform.astype(np.float32, copy=False),
        n_fft=cfg.win_len,
        hop_length=cfg.hop_size,
        win_length=cfg.win_len,
        window=cfg.win,
        center=True,
        pad_mode="reflect",
    )  # [F, T] complex64

    spec = (spec.T * cfg.wnorm).astype(np.complex64)  # [T, F]
    spec_ri = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)  # [T, F, 2]
    return spec_ri[None, ...]  # [1, T, F, 2]


def postprocessing(spec_e: np.ndarray, cfg: STFTConfig) -> np.ndarray:
    """
    spec_e: [1, T, F, 2] float32
    Returns waveform (1D float32, cfg.sr)
    """
    spec_c = spec_e[0].astype(np.float32)  # [T, F, 2]
    spec = (spec_c[..., 0] + 1j * spec_c[..., 1]).T.astype(np.complex64)  # [F, T]

    waveform_e = librosa.istft(
        spec,
        hop_length=cfg.hop_size,
        win_length=cfg.win_len,
        window=cfg.win,
        center=True,
        length=None,
    ).astype(np.float32)

    waveform_e = waveform_e / cfg.wnorm

    # Keep the legacy alignment compensation behavior, scaled by win_len.
    waveform_e = np.concatenate(
        [waveform_e[cfg.win_len * 2 :], np.zeros(cfg.win_len * 2, dtype=np.float32)]
    )

    return waveform_e


# -----------------------------------------------------------------------------
# Audio utilities
# -----------------------------------------------------------------------------

def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    # Average channels to mono
    return np.mean(audio, axis=1)


def ensure_sr(waveform: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return waveform.astype(np.float32, copy=False)
    return librosa.resample(
        waveform.astype(np.float32, copy=False), orig_sr=sr, target_sr=target_sr
    )


def resample_back(waveform_model_sr: np.ndarray, model_sr: int, target_sr: int) -> np.ndarray:
    if target_sr == model_sr:
        return waveform_model_sr
    return librosa.resample(
        waveform_model_sr.astype(np.float32, copy=False),
        orig_sr=model_sr,
        target_sr=target_sr,
    )


def pcm16_safe(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def validate_attn_limit_db(attn_limit_db: float | None) -> float | None:
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
    value = validate_attn_limit_db(attn_limit_db)
    enhanced = np.asarray(spec_enh, dtype=np.float32)
    if value is None:
        return enhanced

    noisy = np.asarray(spec_noisy, dtype=np.float32)
    if noisy.shape != enhanced.shape:
        raise ValueError(
            "spec_noisy and spec_enh must have matching shapes, "
            f"got {noisy.shape} and {enhanced.shape}."
        )

    # The TFLite offline path emits enhanced spectra ~4 hops ahead of the
    # original noisy STFT, so shift the noisy reference to that frame index
    # before blending for attenuation limiting.
    aligned_noisy = np.zeros_like(noisy, dtype=np.float32)
    if noisy.shape[1] > ATTN_LIMIT_NOISY_FRAME_OFFSET:
        aligned_noisy[:, ATTN_LIMIT_NOISY_FRAME_OFFSET:, :, :] = noisy[
            :, :-ATTN_LIMIT_NOISY_FRAME_OFFSET, :, :
        ]

    alpha = float(10.0 ** (-value / 20.0))
    return np.ascontiguousarray(alpha * aligned_noisy + (1.0 - alpha) * enhanced, dtype=np.float32)


# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------

def _load_model_and_cfg(model_name: str) -> tuple[Interpreter, STFTConfig]:
    """Create interpreter and return (interpreter, STFTConfig) for this model."""
    if model_name not in MODEL_CONFIG:
        raise ValueError(
            f"Unknown model '{model_name}'. Add it to MODEL_CONFIG or pass a valid --model_name."
        )

    model_path = TFLITE_DIR / f"{model_name}.tflite"
    if not model_path.exists():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    cfg_dict = MODEL_CONFIG[model_name]
    cfg = make_stft_config(sr=int(cfg_dict["sr"]), win_len=int(cfg_dict["win_len"]))

    # sanity-check: infer expected F from model input and compare
    try:
        input_details = interpreter.get_input_details()
        shape = input_details[0].get("shape", None)
        # Expect [1, T=1, F, 2]
        if shape is not None and len(shape) >= 3:
            F = int(shape[-2])  # ... F, 2
            expected_F = cfg.win_len // 2 + 1
            if F != expected_F:
                raise ValueError(
                    f"Model '{model_name}' input F={F} does not match win_len={cfg.win_len} "
                    f"(expected F={expected_F}). Update MODEL_CONFIG for this model."
                )
    except Exception:
        pass

    return interpreter, cfg


def enhance_file(
    interpreter: Interpreter,
    cfg: STFTConfig,
    in_path: Path,
    out_path: Path,
    model_name: str,
    attn_limit_db: float | None = None,
) -> tuple[int, float, float, float]:
    """Returns (num_frames, total_infer_time_s, avg_frame_ms, rtf)."""
    # Load audio
    audio, sr_in = sf.read(str(in_path), always_2d=False)
    audio = to_mono(audio)
    audio = audio.astype(np.float32, copy=False)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Resample to model SR
    audio_model_sr = ensure_sr(audio, sr_in, cfg.sr)

    # Alignment compensation #1
    audio_pad = np.pad(audio_model_sr, (0, cfg.win_len), mode='constant', constant_values=0)

    # STFT to frames (streaming)
    spec = preprocessing(audio_pad, cfg)  # [1, T, F, 2]
    num_frames = spec.shape[1]

    # Frame-by-frame inference
    outputs = []
    total_infer_time_s = 0.0
    for t in range(num_frames):
        frame = spec[:, t : t + 1]  # [B=1, T=1, F, 2]
        frame = np.ascontiguousarray(frame, dtype=np.float32)

        interpreter.set_tensor(input_details[0]["index"], frame)
        t0 = time.perf_counter()
        interpreter.invoke()
        total_infer_time_s += time.perf_counter() - t0
        y = interpreter.get_tensor(output_details[0]["index"])  # expected [1,1,F,2]
        outputs.append(np.ascontiguousarray(y, dtype=np.float32))

    # Concatenate along time dimension
    spec_e = np.concatenate(outputs, axis=1).astype(np.float32)  # [1, T, F, 2]
    spec_e = apply_attn_limit(spec, spec_e, attn_limit_db)

    # iSTFT to waveform (model SR), then back to original SR for saving
    enhanced_model_sr = postprocessing(spec_e, cfg)
    enhanced = resample_back(enhanced_model_sr, cfg.sr, sr_in)

    # Alignment compensation #2
    enhanced = enhanced[: audio.size]

    # Save as 16-bit PCM WAV, mono, original sample rate
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), pcm16_safe(enhanced), sr_in, subtype="PCM_16")

    avg_frame_ms = (total_infer_time_s / num_frames) * 1000.0 if num_frames > 0 else float("nan")
    frame_duration_s = float(cfg.hop_size) / float(cfg.sr)
    rtf = (
        total_infer_time_s / (num_frames * frame_duration_s)
        if num_frames > 0 and frame_duration_s > 0.0
        else float("nan")
    )
    return num_frames, total_infer_time_s, avg_frame_ms, rtf


def main():
    parser = argparse.ArgumentParser(
        description="Enhance WAV files with a DPDFNet TFLite model (streaming)."
    )
    parser.add_argument(
        "--noisy_dir",
        type=str,
        required=True,
        help="Folder with noisy *.wav files (non-recursive).",
    )
    parser.add_argument(
        "--enhanced_dir",
        type=str,
        required=True,
        help="Output folder for enhanced WAVs.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="dpdfnet8",
        choices=sorted(MODEL_CONFIG.keys()),
        help=(
            "Name of the model to use. The script will automatically use the correct "
            "sample-rate/STFT settings"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker threads. Defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--attn-limit-db",
        "--attn_limit_db",
        dest="attn_limit_db",
        type=float,
        default=None,
        help="Offline-only attenuation limit in dB. Higher values allow stronger denoising.",
    )

    args = parser.parse_args()
    attn_limit_db = validate_attn_limit_db(args.attn_limit_db)
    print_banner(version=None)
    noisy_dir = Path(args.noisy_dir)
    enhanced_dir = Path(args.enhanced_dir)
    model_name = args.model_name

    if not noisy_dir.is_dir():
        print(
            f"ERROR: --noisy_dir does not exist or is not a directory: {noisy_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    wavs = sorted(p for p in noisy_dir.glob("*.wav") if p.is_file())
    if not wavs:
        print(f"No .wav files found in {noisy_dir} (non-recursive).")
        sys.exit(0)

    model_cfg = MODEL_CONFIG[model_name]
    cfg = make_stft_config(sr=int(model_cfg["sr"]), win_len=int(model_cfg["win_len"]))

    print(f"Model: {model_name}")
    print(f"Model SR: {model_cfg['sr']} Hz | win_len: {model_cfg['win_len']} | hop: {model_cfg['win_len']//2}")
    print(f"Attenuation limit: {attn_limit_db if attn_limit_db is not None else 'disabled'}")
    print(f"Input : {noisy_dir}")
    print(f"Output: {enhanced_dir}")

    n_workers = args.workers or (os.cpu_count() or 1)
    print(f"Found {len(wavs)} file(s). Enhancing with {n_workers} worker(s)...\n")

    # Each thread gets its own independent TFLite interpreter (not thread-safe to share).
    _tls = threading.local()

    def _get_interpreter() -> Interpreter:
        interp = getattr(_tls, "interp", None)
        if interp is None:
            interp, _ = _load_model_and_cfg(model_name)
            _tls.interp = interp
        return interp

    def _process(wav: Path) -> tuple[Path, Path, tuple]:
        interp = _get_interpreter()
        out_path = enhanced_dir / (wav.stem + f"_{model_name}.wav")
        result = enhance_file(interp, cfg, wav, out_path, model_name, attn_limit_db=attn_limit_db)
        return wav, out_path, result

    future_to_wav = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for wav in wavs:
            future_to_wav[pool.submit(_process, wav)] = wav

        for future in as_completed(future_to_wav):
            wav = future_to_wav[future]
            exc = future.exception()
            if exc is not None:
                print(f"[SKIP] {wav.name} due to error: {exc}", file=sys.stderr)
                continue
            wav, out_path, (num_frames, total_infer_time_s, avg_frame_ms, rtf) = future.result()
            print(f"[OK] Wrote TFLite enhanced audio: {out_path}")
            print(
                f"[INFO] {wav.name}: frames={num_frames}, "
                f"total={total_infer_time_s:.6f}s, avg_frame={avg_frame_ms:.4f}ms, rtf={rtf:.6f}"
            )

    print("\nProcessing complete. Outputs saved in:", enhanced_dir)


if __name__ == "__main__":
    main()
