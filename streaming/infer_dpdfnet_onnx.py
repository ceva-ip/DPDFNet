import argparse
import os
import sys
import time
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import onnxruntime as ort
import soundfile
import torch

from banner import print_banner
from streaming.dpdfnet import DPDFNet
from streaming.dpdfnet_48khz_hr import DPDFNet48HR

MODEL_NAME_TO_DPRNN_NUM_BLOCKS = {
    "baseline": 0,
    "dpdfnet2": 2,
    "dpdfnet4": 4,
    "dpdfnet8": 8,
    "dpdfnet2_48khz_hr": 2,
}


def build_dpdfnet(dprnn_num_blocks: int) -> DPDFNet:
    model = DPDFNet(
        conv_kernel_inp=(3, 3),
        conv_ch=64,
        enc_gru_dim=256,
        erb_dec_gru_dim=256,
        df_dec_gru_dim=256,
        enc_lin_groups=32,
        lin_groups=16,
        upsample_conv_type="subpixel",
        group_linear_type="loop",
        point_wise_type="cnn",
        separable_first_conv=True,
        dprnn_num_blocks=dprnn_num_blocks,
    )
    model.eval()
    return model


def build_dpdfnet48hr(dprnn_num_blocks: int) -> DPDFNet48HR:
    model = DPDFNet48HR(
        conv_kernel_inp=(3, 3),
        conv_ch=64,
        enc_gru_dim=256,
        erb_dec_gru_dim=256,
        df_dec_gru_dim=256,
        enc_lin_groups=32,
        lin_groups=16,
        upsample_conv_type="subpixel",
        group_linear_type="loop",
        point_wise_type="cnn",
        separable_first_conv=True,
        dprnn_num_blocks=dprnn_num_blocks,
    )
    model.eval()
    return model


def normalize_model_type(model_type: str) -> str:
    normalized = model_type.lower().replace("-", "_")
    if normalized == "dpdfnet":
        return "dpdfnet"
    if normalized in ("dpdfnet48hr", "dpdfnet_48khz_hr"):
        return "dpdfnet48hr"
    raise ValueError(f"Unsupported model type: {model_type}")


def infer_model_type_from_onnx_name(onnx_path: Path) -> str:
    model_name = onnx_path.stem.lower().replace("-", "_")
    if "48khz" in model_name or "48k" in model_name or "48hr" in model_name:
        return "dpdfnet48hr"
    if "baseline" in model_name or "dpdfnet" in model_name:
        return "dpdfnet"
    raise ValueError(
        f"Could not infer model type from ONNX filename: {onnx_path.name}. "
        "Use --model-type to set it explicitly."
    )


def infer_dprnn_num_blocks_from_onnx_name(onnx_path: Path) -> int:
    model_name = onnx_path.stem.lower().replace("-", "_")
    if model_name in MODEL_NAME_TO_DPRNN_NUM_BLOCKS:
        return MODEL_NAME_TO_DPRNN_NUM_BLOCKS[model_name]
    raise ValueError(
        f"Could not infer DPRNN blocks from ONNX filename: {onnx_path.name}. "
        f"Expected one of: {sorted(MODEL_NAME_TO_DPRNN_NUM_BLOCKS)}"
    )


def infer_model_type_from_session(session: ort.InferenceSession) -> str:
    spec_shape = session.get_inputs()[0].shape
    freq_bins = spec_shape[-2] if len(spec_shape) >= 2 else None

    if isinstance(freq_bins, int):
        if freq_bins == 161:
            return "dpdfnet"
        if freq_bins == 481:
            return "dpdfnet48hr"

    raise ValueError(
        "Could not infer model type from ONNX graph input shape. "
        f"Got spec input shape={spec_shape}. Use --model-type explicitly."
    )


def build_model(model_type: str, dprnn_num_blocks: int) -> tuple[Union[DPDFNet, DPDFNet48HR], int]:
    normalized = normalize_model_type(model_type)
    if normalized == "dpdfnet":
        return build_dpdfnet(dprnn_num_blocks=dprnn_num_blocks), 16000
    if normalized == "dpdfnet48hr":
        return build_dpdfnet48hr(dprnn_num_blocks=dprnn_num_blocks), 48000
    raise ValueError(f"Unsupported model type: {model_type}")


def build_session(onnx_path: Path, providers_priority: list[str]) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    available = set(ort.get_available_providers())
    providers = [provider for provider in providers_priority if provider in available]
    if not providers:
        raise RuntimeError(
            f"None of requested providers are available. requested={providers_priority}, available={sorted(available)}"
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

    state = init_state.astype(np.float32, copy=True)
    spec_e_list = []
    total_infer_time_s = 0.0

    for t in range(spec_r.shape[1]):
        spec_t = spec_r[:, t:t + 1, :, :].astype(np.float32, copy=False)
        t0 = time.perf_counter()
        spec_e_t, state = session.run([out_spec, out_state], {in_spec: spec_t, in_state: state})
        total_infer_time_s += time.perf_counter() - t0
        spec_e_list.append(spec_e_t)

    spec_e = np.concatenate(spec_e_list, axis=1)
    return spec_e, state, total_infer_time_s


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
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


def fit_length(waveform: np.ndarray, target_len: int) -> np.ndarray:
    if waveform.size >= target_len:
        return waveform[:target_len]
    return np.pad(waveform, (0, target_len - waveform.size), mode="constant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run concise DPDFNet ONNX streaming inference.")
    parser.add_argument("--onnx", type=Path, required=True, help="Path to ONNX model.")
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
        "--model-type",
        choices=["auto", "dpdfnet", "dpdfnet48hr", "dpdfnet_48khz_hr"],
        default="auto",
        help="Model frontend used for STFT/iSTFT and initial state. Default: auto (from ONNX filename).",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider"],
        help="Execution providers in priority order.",
    )
    return parser.parse_args()


def enhance_file_onnx(
    session: ort.InferenceSession,
    model: Union[DPDFNet, DPDFNet48HR],
    expected_sr: int,
    input_wav: Path,
    output_wav: Path,
) -> tuple[int, tuple[int, ...], float, float, float]:
    waveform, sr_in = soundfile.read(str(input_wav))
    waveform = to_mono(waveform).astype(np.float32, copy=False)
    waveform_model_sr = ensure_sr(waveform, sr_in, expected_sr)

    waveform_t = torch.tensor(waveform_model_sr, dtype=torch.float32).flatten().unsqueeze(0)
    spec = model.apply_stft(waveform_t)
    spec_r_np = torch.view_as_real(spec).to(dtype=torch.float32).cpu().numpy()
    init_state = model.initial_state(dtype=torch.float32).cpu().numpy()

    spec_e_np, state_out_np, total_infer_time_s = run_onnx_streaming(session, spec_r_np, init_state)
    spec_e_t = torch.tensor(spec_e_np, dtype=torch.float32)
    waveform_e_model_sr = model.apply_istft(torch.view_as_complex(spec_e_t)).squeeze(0).cpu().numpy()
    waveform_e = resample_back(waveform_e_model_sr, expected_sr, sr_in)
    waveform_e = fit_length(waveform_e, waveform.size)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(output_wav), waveform_e, sr_in)

    num_frames = int(spec_r_np.shape[1])
    avg_frame_ms = (total_infer_time_s / num_frames) * 1000.0 if num_frames > 0 else float("nan")
    frame_duration_s = float(model.stft.hop) / float(expected_sr)
    rtf = (
        total_infer_time_s / (num_frames * frame_duration_s)
        if num_frames > 0 and frame_duration_s > 0.0
        else float("nan")
    )
    return num_frames, tuple(state_out_np.shape), total_infer_time_s, avg_frame_ms, rtf


def main() -> None:
    args = parse_args()
    print_banner(version="1.0.0")

    onnx_path = args.onnx.expanduser().resolve()
    onnx_suffix = onnx_path.stem
    noisy_dir = args.noisy_dir.expanduser().resolve()
    enhanced_dir = args.enhanced_dir.expanduser().resolve()

    if not noisy_dir.is_dir():
        print(f"ERROR: --noisy_dir does not exist or is not a directory: {noisy_dir}", file=sys.stderr)
        sys.exit(1)

    wavs = sorted(p for p in noisy_dir.glob("*.wav") if p.is_file())
    if not wavs:
        print(f"No .wav files found in {noisy_dir} (non-recursive).")
        sys.exit(0)

    if args.model_type == "auto":
        try:
            model_type = infer_model_type_from_onnx_name(onnx_path)
            print(f"[INFO] Auto-detected model type from ONNX name '{onnx_path.name}': {model_type}")
        except ValueError:
            session_for_auto = build_session(onnx_path, providers_priority=args.providers)
            model_type = infer_model_type_from_session(session_for_auto)
            print(
                "[INFO] Auto-detected model type from ONNX input shape "
                f"{session_for_auto.get_inputs()[0].shape}: {model_type}"
            )
    else:
        model_type = normalize_model_type(args.model_type)

    dprnn_num_blocks = infer_dprnn_num_blocks_from_onnx_name(onnx_path)
    model, expected_sr = build_model(model_type=model_type, dprnn_num_blocks=dprnn_num_blocks)
    session = build_session(onnx_path, providers_priority=args.providers)

    print(f"[INFO] ONNX Runtime version: {ort.__version__}")
    print(f"[INFO] Host CPU cores (os.cpu_count): {os.cpu_count()}")
    print(f"[INFO] Active providers: {session.get_providers()}")
    print(f"[INFO] Model type: {model_type}")
    print(f"[INFO] DPRNN blocks: {dprnn_num_blocks}")
    print(f"[INFO] Model SR: {expected_sr} Hz")
    print(f"Input : {noisy_dir}")
    print(f"Output: {enhanced_dir}")
    print(f"Found {len(wavs)} file(s). Enhancing...\n")

    for wav in wavs:
        output_wav = enhanced_dir / (wav.stem + f"_{onnx_suffix}.wav")
        try:
            num_frames, state_shape, total_infer_time_s, avg_frame_ms, rtf = enhance_file_onnx(
                session=session,
                model=model,
                expected_sr=expected_sr,
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
