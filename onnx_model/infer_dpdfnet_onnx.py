import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile

from banner import print_banner

ONNX_DIR = Path("./model_zoo/onnx")

MODEL_SAMPLE_RATE_BY_NAME = {
    "baseline": 16000,
    "dpdfnet2": 16000,
    "dpdfnet4": 16000,
    "dpdfnet8": 16000,
    "dpdfnet2_48khz_hr": 48000,
}


@dataclass(frozen=True)
class StftConfig:
    win_len: int
    hop_size: int
    window: np.ndarray
    wnorm: float
def infer_win_len(session: ort.InferenceSession, default_sr: int) -> int:
    spec_shape = session.get_inputs()[0].shape
    freq_bins = spec_shape[-2] if len(spec_shape) >= 2 else None
    if isinstance(freq_bins, int) and freq_bins > 1:
        return int((freq_bins - 1) * 2)
    return int(round(default_sr * 0.02))


def resolve_state_path(onnx_path: Path) -> Path:
    resolved = onnx_path.with_name(f"{onnx_path.stem}_state.npz")
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Initial state file not found: {resolved}. "
            "Expected <model>_state.npz next to the ONNX file."
        )
    return resolved


def load_initial_state(session: ort.InferenceSession, state_path: Path) -> np.ndarray:
    with np.load(state_path) as data:
        if "init_state" not in data:
            raise ValueError(f"Missing 'init_state' key in state file: {state_path}")
        init_state = np.ascontiguousarray(data["init_state"].astype(np.float32, copy=False))

    if len(session.get_inputs()) < 2:
        raise ValueError("Expected streaming ONNX model with two inputs: (spec, state).")

    expected_shape = session.get_inputs()[1].shape
    if len(expected_shape) != init_state.ndim:
        raise ValueError(
            f"Initial state rank mismatch: expected={expected_shape}, actual={init_state.shape}"
        )

    for exp_dim, act_dim in zip(expected_shape, init_state.shape):
        if isinstance(exp_dim, int) and exp_dim != act_dim:
            raise ValueError(
                f"Initial state shape mismatch: expected={expected_shape}, actual={init_state.shape}"
            )

    return init_state


def build_session(onnx_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    available = set(ort.get_available_providers())
    providers = ["CPUExecutionProvider"]
    if "CPUExecutionProvider" not in available:
        raise RuntimeError(
            "CPUExecutionProvider is not available. "
            f"available={sorted(available)}"
        )

    return ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers)


def run_onnx_streaming(
    session: ort.InferenceSession,
    spec_r: np.ndarray,
    init_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    in_spec = session.get_inputs()[0].name
    in_state = session.get_inputs()[1].name
    out_spec = session.get_outputs()[0].name
    out_state = session.get_outputs()[1].name

    state = np.ascontiguousarray(init_state.astype(np.float32, copy=True))
    spec_e_list = []
    total_infer_time_s = 0.0

    for t in range(spec_r.shape[1]):
        spec_t = np.ascontiguousarray(spec_r[:, t : t + 1, :, :], dtype=np.float32)
        t0 = time.perf_counter()
        spec_e_t, state = session.run([out_spec, out_state], {in_spec: spec_t, in_state: state})
        total_infer_time_s += time.perf_counter() - t0
        spec_e_list.append(np.ascontiguousarray(spec_e_t, dtype=np.float32))

    spec_e = np.concatenate(spec_e_list, axis=1)
    return spec_e, state, total_infer_time_s


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim != 2:
        raise ValueError(f"Expected mono/stereo audio, got shape {audio.shape}")
    return np.mean(audio, axis=1, dtype=np.float32)


def ensure_sr(waveform: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return waveform.astype(np.float32, copy=False)
    return librosa.resample(
        waveform.astype(np.float32, copy=False),
        orig_sr=sr,
        target_sr=target_sr,
    ).astype(np.float32, copy=False)


def resample_back(waveform_model_sr: np.ndarray, model_sr: int, target_sr: int) -> np.ndarray:
    if target_sr == model_sr:
        return waveform_model_sr
    return librosa.resample(
        waveform_model_sr.astype(np.float32, copy=False),
        orig_sr=model_sr,
        target_sr=target_sr,
    ).astype(np.float32, copy=False)


def fit_length(waveform: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if x.shape[0] == target_len:
        return x
    if x.shape[0] > target_len:
        return x[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[: x.shape[0]] = x
    return out


def vorbis_window(window_len: int) -> np.ndarray:
    window_size_h = window_len / 2
    indices = np.arange(window_len)
    s = np.sin(0.5 * np.pi * (indices + 0.5) / window_size_h)
    return np.sin(0.5 * np.pi * s * s).astype(np.float32)


def get_wnorm(window_len: int, frame_size: int) -> float:
    return 1.0 / (window_len**2 / (2 * frame_size))


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
    return np.ascontiguousarray(spec_ri[None, ...], dtype=np.float32)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run concise DPDFNet ONNX streaming inference.")
    parser.add_argument(
        "--noisy_dir",
        type=Path,
        required=True,
        help="Folder with noisy *.wav files (non-recursive).",
    )
    parser.add_argument(
        "--enhanced_dir",
        type=Path,
        required=True,
        help="Output folder for enhanced WAVs.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="dpdfnet8",
        choices=sorted(MODEL_SAMPLE_RATE_BY_NAME.keys()),
        help="Model name. The ONNX model will be loaded from model_zoo/onnx/<model_name>.onnx",
    )
    return parser.parse_args()


def enhance_file_onnx(
    session: ort.InferenceSession,
    init_state: np.ndarray,
    expected_sr: int,
    win_len: int,
    input_wav: Path,
    output_wav: Path,
) -> tuple[int, tuple[int, ...], float, float, float]:
    waveform, sr_in = soundfile.read(str(input_wav))
    waveform = to_mono(np.asarray(waveform, dtype=np.float32))
    waveform_model_sr = ensure_sr(waveform, sr_in, expected_sr)

    cfg = make_stft_config(win_len)
    waveform_padded = np.pad(waveform_model_sr, (0, cfg.win_len), mode="constant")
    spec_r_np = preprocess_waveform(waveform_padded, cfg)

    spec_e_np, state_out_np, total_infer_time_s = run_onnx_streaming(session, spec_r_np, init_state)
    waveform_e_model_sr = postprocess_spec(spec_e_np, cfg)
    waveform_e = resample_back(waveform_e_model_sr, expected_sr, sr_in)
    waveform_e = fit_length(waveform_e, waveform.size)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(output_wav), waveform_e, sr_in)

    num_frames = int(spec_r_np.shape[1])
    avg_frame_ms = (total_infer_time_s / num_frames) * 1000.0 if num_frames > 0 else float("nan")
    frame_duration_s = float(cfg.hop_size) / float(expected_sr)
    rtf = (
        total_infer_time_s / (num_frames * frame_duration_s)
        if num_frames > 0 and frame_duration_s > 0.0
        else float("nan")
    )
    return num_frames, tuple(state_out_np.shape), total_infer_time_s, avg_frame_ms, rtf


def main() -> None:
    args = parse_args()
    print_banner(version=None)

    model_name = args.model_name
    onnx_path = (ONNX_DIR / f"{model_name}.onnx").resolve()
    if not onnx_path.is_file():
        print(f"ERROR: ONNX model not found for --model_name='{model_name}': {onnx_path}", file=sys.stderr)
        sys.exit(1)

    expected_sr = MODEL_SAMPLE_RATE_BY_NAME[model_name]
    state_path = resolve_state_path(onnx_path)
    onnx_suffix = model_name
    noisy_dir = args.noisy_dir.expanduser().resolve()
    enhanced_dir = args.enhanced_dir.expanduser().resolve()

    if not noisy_dir.is_dir():
        print(f"ERROR: --noisy_dir does not exist or is not a directory: {noisy_dir}", file=sys.stderr)
        sys.exit(1)

    wavs = sorted(p for p in noisy_dir.glob("*.wav") if p.is_file())
    if not wavs:
        print(f"No .wav files found in {noisy_dir} (non-recursive).")
        sys.exit(0)

    session = build_session(onnx_path)

    win_len = infer_win_len(session, expected_sr)
    init_state = load_initial_state(session, state_path)

    print(f"[INFO] ONNX Runtime version: {ort.__version__}")
    print(f"[INFO] Host CPU cores (os.cpu_count): {os.cpu_count()}")
    print(f"[INFO] Active providers: {session.get_providers()}")
    print(f"[INFO] Model name: {model_name}")
    print(f"[INFO] Model SR: {expected_sr} Hz")
    print(f"[INFO] STFT win_len: {win_len}, hop: {win_len // 2}")
    print(f"[INFO] Initial state: {state_path}")
    print(f"[INFO] Initial state shape: {tuple(init_state.shape)}")
    print(f"Input : {noisy_dir}")
    print(f"Output: {enhanced_dir}")
    print(f"Found {len(wavs)} file(s). Enhancing...\n")

    for wav in wavs:
        output_wav = enhanced_dir / (wav.stem + f"_{onnx_suffix}.wav")
        try:
            num_frames, state_shape, total_infer_time_s, avg_frame_ms, rtf = enhance_file_onnx(
                session=session,
                init_state=init_state,
                expected_sr=expected_sr,
                win_len=win_len,
                input_wav=wav,
                output_wav=output_wav,
            )
        except Exception as e:
            print(f"[SKIP] {wav.name} due to error: {e}", file=sys.stderr)
            continue

        print(f"[OK] Wrote ONNX enhanced audio: {output_wav}")
        print(f"[INFO] {wav.name}: frames={num_frames}, final_state_shape={state_shape}")
        print(
            f"[INFO] {wav.name}: total={total_infer_time_s:.6f}s, "
            f"avg_frame={avg_frame_ms:.4f}ms, rtf={rtf:.6f}"
        )

    print("\nProcessing complete. Outputs saved in:", enhanced_dir)


if __name__ == "__main__":
    main()
