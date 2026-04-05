from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from .models import (
    DEFAULT_MODEL,
    available_model_entries,
    download_model,
    download_models,
    resolve_model,
)


def available_models(
) -> List[Dict[str, Any]]:
    return available_model_entries()


def download(
    model: Optional[str] = None,
    *,
    force: bool = False,
    quiet: bool = False,
    verbose: bool = False,
) -> Union[Path, Dict[str, Path]]:
    if quiet and verbose:
        raise ValueError("quiet=True and verbose=True are mutually exclusive.")

    notifier = (lambda _message: None) if quiet else None

    if model is None:
        resolved_all = download_models(
            models=None,
            force=force,
            verbose=verbose,
            notifier=notifier,
        )
        return {item.info.name: item.onnx_path.parent for item in resolved_all}

    resolved = download_model(
        model=model,
        force=force,
        verbose=verbose,
        notifier=notifier,
    )
    return resolved.onnx_path.parent


def enhance(
    audio: np.ndarray,
    sample_rate: int,
    *,
    model: str = DEFAULT_MODEL,
    onnx_path: Optional[Union[str, Path]] = None,
    attn_limit_db: Optional[float] = None,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    from .audio import (
        apply_attn_limit,
        ensure_sample_rate,
        fit_length,
        make_stft_config,
        postprocess_spec,
        preprocess_waveform,
        to_mono,
    )
    from .onnx_backend import build_runtime_model, infer_win_len

    waveform = to_mono(np.asarray(audio, dtype=np.float32))
    sr_in = int(sample_rate)

    resolved = resolve_model(
        model=model,
        onnx_path=onnx_path,
        auto_download=True,
        verbose=verbose,
    )
    runtime = build_runtime_model(resolved.onnx_path)

    waveform_model_sr = ensure_sample_rate(waveform, sr_in, resolved.info.sample_rate)
    win_len = infer_win_len(runtime.session, resolved.info.sample_rate)
    cfg = make_stft_config(win_len)

    # Keep alignment behavior from the original scripts.
    waveform_padded = np.pad(waveform_model_sr, (0, cfg.win_len), mode="constant")
    spec_r = preprocess_waveform(waveform_padded, cfg)

    state = runtime.init_state.copy()
    frames: list[np.ndarray] = []
    total_frames = int(spec_r.shape[1])
    if progress_callback is not None:
        progress_callback(0, total_frames)
    for t in range(total_frames):
        spec_t = np.ascontiguousarray(spec_r[:, t : t + 1, :, :], dtype=np.float32)
        spec_e_t, state = runtime.session.run(
            [runtime.out_spec_name, runtime.out_state_name],
            {runtime.in_spec_name: spec_t, runtime.in_state_name: state},
        )
        frames.append(np.ascontiguousarray(spec_e_t, dtype=np.float32))
        if progress_callback is not None:
            progress_callback(t + 1, total_frames)

    if not frames:
        return waveform.copy()

    spec_e = np.concatenate(frames, axis=1)
    spec_e = apply_attn_limit(spec_r, spec_e, attn_limit_db)
    enhanced_model_sr = postprocess_spec(spec_e, cfg)
    enhanced = ensure_sample_rate(enhanced_model_sr, resolved.info.sample_rate, sr_in)
    return fit_length(enhanced, waveform.shape[0]).astype(np.float32, copy=False)


def _enhance_with_runtime(
    audio: np.ndarray,
    sample_rate: int,
    *,
    runtime,
    model_sample_rate: int,
    attn_limit_db: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Like `enhance()` but skips model resolution/loading; uses a pre-built RuntimeModel."""
    from .audio import (
        apply_attn_limit,
        ensure_sample_rate,
        fit_length,
        make_stft_config,
        postprocess_spec,
        preprocess_waveform,
        to_mono,
    )
    from .onnx_backend import infer_win_len

    waveform = to_mono(np.asarray(audio, dtype=np.float32))
    sr_in = int(sample_rate)

    waveform_model_sr = ensure_sample_rate(waveform, sr_in, model_sample_rate)
    win_len = infer_win_len(runtime.session, model_sample_rate)
    cfg = make_stft_config(win_len)

    waveform_padded = np.pad(waveform_model_sr, (0, cfg.win_len), mode="constant")
    spec_r = preprocess_waveform(waveform_padded, cfg)

    state = runtime.init_state.copy()
    frames: list[np.ndarray] = []
    total_frames = int(spec_r.shape[1])
    if progress_callback is not None:
        progress_callback(0, total_frames)
    for t in range(total_frames):
        spec_t = np.ascontiguousarray(spec_r[:, t : t + 1, :, :], dtype=np.float32)
        spec_e_t, state = runtime.session.run(
            [runtime.out_spec_name, runtime.out_state_name],
            {runtime.in_spec_name: spec_t, runtime.in_state_name: state},
        )
        frames.append(np.ascontiguousarray(spec_e_t, dtype=np.float32))
        if progress_callback is not None:
            progress_callback(t + 1, total_frames)

    if not frames:
        return waveform.copy()

    spec_e = np.concatenate(frames, axis=1)
    spec_e = apply_attn_limit(spec_r, spec_e, attn_limit_db)
    enhanced_model_sr = postprocess_spec(spec_e, cfg)
    enhanced = ensure_sample_rate(enhanced_model_sr, model_sample_rate, sr_in)
    return fit_length(enhanced, waveform.shape[0]).astype(np.float32, copy=False)


def _enhance_file_with_runtime(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    runtime,
    model_sample_rate: int,
    attn_limit_db: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """Like `enhance_file()` but uses a pre-built RuntimeModel to skip model loading."""
    import soundfile as sf

    from .audio import pcm16_safe

    in_path = Path(input_path).expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    audio, sr = _read_audio(in_path)
    enhanced = _enhance_with_runtime(
        audio=audio,
        sample_rate=int(sr),
        runtime=runtime,
        model_sample_rate=model_sample_rate,
        attn_limit_db=attn_limit_db,
        progress_callback=progress_callback,
    )

    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), pcm16_safe(enhanced), int(sr), subtype="PCM_16")
    return out_path


# Extensions handled natively by soundfile (libsndfile)
_SF_EXTENSIONS: frozenset[str] = frozenset({".wav", ".flac", ".ogg", ".aiff", ".aif", ".au", ".snd"})
# Extensions that require pydub + ffmpeg
_PYDUB_EXTENSIONS: frozenset[str] = frozenset({".mp3", ".m4a", ".aac", ".wma", ".opus"})
# All supported input extensions
SUPPORTED_EXTENSIONS: frozenset[str] = _SF_EXTENSIONS | _PYDUB_EXTENSIONS


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return (audio float32, sample_rate) for any supported audio format."""
    suffix = path.suffix.lower()
    if suffix in _SF_EXTENSIONS:
        import soundfile as sf
        audio, sr = sf.read(str(path), always_2d=False)
        return np.asarray(audio, dtype=np.float32), int(sr)
    if suffix in _PYDUB_EXTENSIONS:
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                f"Reading {suffix!r} files requires the 'pydub' package and ffmpeg.\n"
                f"Install them with:  pip install 'dpdfnet[mp3]'\n"
                f"and ensure ffmpeg is available on your PATH."
            ) from None
        seg = AudioSegment.from_file(str(path))
        sr = seg.frame_rate
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        samples /= float(1 << (seg.sample_width * 8 - 1))
        if seg.channels > 1:
            samples = samples.reshape(-1, seg.channels)
        return samples, sr
    supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    raise ValueError(
        f"Unsupported audio format {suffix!r} for file: {path}\n"
        f"Supported extensions: {supported}"
    )


def enhance_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    model: str = DEFAULT_MODEL,
    onnx_path: Optional[Union[str, Path]] = None,
    attn_limit_db: Optional[float] = None,
    verbose: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Path:
    import soundfile as sf

    from .audio import pcm16_safe

    in_path = Path(input_path).expanduser().resolve()
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    audio, sr = _read_audio(in_path)
    enhanced = enhance(
        audio=audio,
        sample_rate=int(sr),
        model=model,
        onnx_path=onnx_path,
        attn_limit_db=attn_limit_db,
        verbose=verbose,
        progress_callback=progress_callback,
    )

    if output_path is None:
        out_path = in_path.with_name(f"{in_path.stem}_enhanced.wav")
    else:
        out_path = Path(output_path).expanduser().resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), pcm16_safe(enhanced), int(sr), subtype="PCM_16")
    return out_path
