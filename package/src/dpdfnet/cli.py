from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import metadata
import os
from pathlib import Path
import sys
import threading
from typing import Callable, List, Optional

from tqdm import tqdm
from .banner import print_banner
from .models import DEFAULT_MODEL, get_cache_model_dir, get_model_info, supported_models


def _build_frame_progress_callback(
    bar: tqdm,
) -> Callable[[int, int], None]:
    last_done = 0

    def _callback(done: int, total: int) -> None:
        nonlocal last_done
        if bar.total != total:
            bar.total = total
            bar.refresh()
        delta = max(0, done - last_done)
        if delta:
            bar.update(delta)
        last_done = done

    return _callback


def _version_string() -> str:
    try:
        return f"dpdfnet {metadata.version('dpdfnet')}"
    except metadata.PackageNotFoundError:
        return "dpdfnet (local)"


def _add_model_resolution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=supported_models(),
        help="Model name to run.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose model-resolution/download logs.",
    )


def _add_attn_limit_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--attn-limit-db",
        "--attn_limit_db",
        dest="attn_limit_db",
        type=float,
        default=None,
        help="Offline-only attenuation limit in dB. Higher values allow stronger denoising.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dpdfnet",
        description="DPDFNet CPU-only ONNX speech enhancement toolkit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=_version_string(),
    )

    subparsers = parser.add_subparsers(dest="command")

    p_models = subparsers.add_parser(
        "models",
        help="List supported models and local availability.",
    )

    p_enhance = subparsers.add_parser(
        "enhance",
        help="Enhance a single audio file (.wav, .flac, .mp3, .ogg, …).",
    )
    p_enhance.add_argument(
        "input", type=Path,
        help="Input audio file (.wav, .flac, .mp3, .ogg, and more).",
    )
    p_enhance.add_argument("output", type=Path, help="Output wav file path.")
    _add_attn_limit_arg(p_enhance)
    _add_model_resolution_args(p_enhance)

    p_enhance_dir = subparsers.add_parser(
        "enhance-dir",
        help="Enhance all supported audio files from one directory (non-recursive).",
    )
    p_enhance_dir.add_argument(
        "input_dir", type=Path,
        help="Input directory containing audio files (.wav, .flac, .mp3, .ogg, …).",
    )
    p_enhance_dir.add_argument("output_dir", type=Path, help="Output directory.")
    p_enhance_dir.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel workers (default: CPU count).",
    )
    _add_attn_limit_arg(p_enhance_dir)
    _add_model_resolution_args(p_enhance_dir)

    p_download = subparsers.add_parser(
        "download",
        help="Download all models by default, or a single model if provided.",
    )
    p_download.add_argument(
        "model",
        nargs="?",
        choices=supported_models(),
        default=None,
        help="Optional model name to download. If omitted, all models are fetched.",
    )
    p_download.add_argument(
        "--model",
        dest="model_flag",
        choices=supported_models(),
        default=None,
        help=argparse.SUPPRESS,
    )
    p_download.add_argument(
        "--force",
        "--refresh",
        action="store_true",
        help="Force re-download even if files are already cached.",
    )
    p_download_verbosity = p_download.add_mutually_exclusive_group()
    p_download_verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress download progress messages.",
    )
    p_download_verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose download logs.",
    )

    return parser


def _print_model_table() -> int:
    from .api import available_models

    rows = available_models()

    headers = ["Model", "Sample Rate", "Ready", "Cached", "Description"]
    col_keys = ["name", "sample_rate", "ready", "cached", "description"]

    def fmt(row: dict, key: str) -> str:
        v = row[key]
        if key == "sample_rate":
            return f"{v // 1000} kHz"
        if isinstance(v, bool):
            return "yes" if v else "no"
        return str(v)

    table = [[fmt(r, k) for k in col_keys] for r in rows]
    col_widths = [max(len(h), *(len(r[i]) for r in table)) for i, h in enumerate(headers)]

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_row = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    print(f"\n  Cache dir: {get_cache_model_dir().resolve()}\n")
    print(sep)
    print(header_row)
    print(sep)
    for r in table:
        print("| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) + " |")
    print(sep)
    print()
    return 0


def _run_enhance(args: argparse.Namespace) -> int:
    from .api import enhance_file

    info = get_model_info(args.model)
    print_banner(
        model_name=info.name,
        sample_rate=info.sample_rate,
        description=info.description,
    )

    with tqdm(
        total=0,
        unit="frame",
        desc="Enhancing",
        dynamic_ncols=True,
        file=sys.stderr,
    ) as progress:
        enhance_file(
            input_path=args.input,
            output_path=args.output,
            model=args.model,
            attn_limit_db=args.attn_limit_db,
            verbose=args.verbose,
            progress_callback=_build_frame_progress_callback(progress),
        )
    print(f"Wrote enhanced audio: {Path(args.output).expanduser().resolve()}")
    return 0


def _run_enhance_dir(args: argparse.Namespace) -> int:
    from .api import SUPPORTED_EXTENSIONS, _enhance_file_with_runtime
    from .models import resolve_model
    from .onnx_backend import build_runtime_model

    info = get_model_info(args.model)
    print_banner(
        model_name=info.name,
        sample_rate=info.sample_rate,
        description=info.description,
    )

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    audio_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not audio_files:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FileNotFoundError(
            f"No supported audio files found in {input_dir}\n"
            f"Supported extensions: {supported}"
        )

    resolved = resolve_model(model=args.model, auto_download=True, verbose=args.verbose)
    n_workers = args.workers or (os.cpu_count() or 1)

    # Each thread gets its own ORT session to avoid lock contention.
    _tls = threading.local()

    def _get_runtime():
        rt = getattr(_tls, "runtime", None)
        if rt is None:
            rt = build_runtime_model(resolved.onnx_path)
            _tls.runtime = rt
        return rt

    output_dir.mkdir(parents=True, exist_ok=True)
    _total_lock = threading.Lock()

    with tqdm(
        total=len(audio_files),
        unit="file",
        desc="Files",
        dynamic_ncols=True,
        file=sys.stderr,
    ) as files_progress:
        with tqdm(
            total=0,
            unit="frame",
            desc="Frames",
            dynamic_ncols=True,
            file=sys.stderr,
        ) as frames_progress:

            def _make_callback(wav_path: Path):
                last_done = 0

                def _callback(done: int, total: int) -> None:
                    nonlocal last_done
                    if done == 0:
                        with _total_lock:
                            frames_progress.total = (frames_progress.total or 0) + total
                            frames_progress.refresh()
                        last_done = 0
                        return
                    delta = max(0, done - last_done)
                    if delta:
                        frames_progress.update(delta)
                    last_done = done

                return _callback

            def _process(wav_path: Path) -> Path:
                out_path = output_dir / f"{wav_path.stem}_enhanced.wav"
                return _enhance_file_with_runtime(
                    input_path=wav_path,
                    output_path=out_path,
                    runtime=_get_runtime(),
                    model_sample_rate=resolved.info.sample_rate,
                    attn_limit_db=args.attn_limit_db,
                    progress_callback=_make_callback(wav_path),
                )

            future_to_path = {}
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for wav_path in audio_files:
                    future_to_path[pool.submit(_process, wav_path)] = wav_path

                errors = []
                for future in as_completed(future_to_path):
                    wav_path = future_to_path[future]
                    exc = future.exception()
                    if exc is not None:
                        errors.append((wav_path, exc))
                    files_progress.update(1)
                    files_progress.set_postfix_str(wav_path.name)

            if errors:
                msgs = "\n".join(f"  {p}: {e}" for p, e in errors)
                raise RuntimeError(f"Errors during processing:\n{msgs}")

    return 0


def _run_download(args: argparse.Namespace) -> int:
    from .api import download

    if args.model is not None and args.model_flag is not None and args.model != args.model_flag:
        raise ValueError("Conflicting model names provided in positional argument and --model.")

    model = args.model if args.model is not None else args.model_flag
    destination = download(
        model=model,
        force=args.force,
        quiet=args.quiet,
        verbose=args.verbose,
    )
    if isinstance(destination, dict):
        print("Downloaded models:")
        for model_name, model_path in destination.items():
            print(f"- {model_name}: {model_path}")
    else:
        model_name = model if model is not None else "<unknown>"
        print(f"Downloaded '{model_name}' to: {destination}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "models":
            return _print_model_table()
        if args.command == "enhance":
            return _run_enhance(args)
        if args.command == "enhance-dir":
            return _run_enhance_dir(args)
        if args.command == "download":
            return _run_download(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
