from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import sys
from typing import Callable, List, Optional

from tqdm import tqdm
from .models import DEFAULT_MODEL, get_cache_model_dir, supported_models


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
        help="Enhance a single wav file.",
    )
    p_enhance.add_argument("input", type=Path, help="Input wav file path.")
    p_enhance.add_argument("output", type=Path, help="Output wav file path.")
    _add_model_resolution_args(p_enhance)

    p_enhance_dir = subparsers.add_parser(
        "enhance-dir",
        help="Enhance all .wav files from one directory (non-recursive).",
    )
    p_enhance_dir.add_argument("input_dir", type=Path, help="Input directory.")
    p_enhance_dir.add_argument("output_dir", type=Path, help="Output directory.")
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
    print(f"cache_dir={get_cache_model_dir().resolve()}")
    for row in rows:
        print(
            f"{row['name']}: sr={row['sample_rate']}Hz, "
            f"ready={row['ready']}, "
            f"onnx_found={row['onnx_found']}, state_found={row['state_found']}, "
            f"cached={row['cached']}"
        )
    return 0


def _run_enhance(args: argparse.Namespace) -> int:
    from .api import enhance_file

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
            verbose=args.verbose,
            progress_callback=_build_frame_progress_callback(progress),
        )
    print(f"Wrote enhanced audio: {Path(args.output).expanduser().resolve()}")
    return 0


def _run_enhance_dir(args: argparse.Namespace) -> int:
    from .api import enhance_file

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    wav_files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav"])
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tqdm(
        total=len(wav_files),
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
            for wav_path in wav_files:
                out_path = output_dir / f"{wav_path.stem}_enhanced.wav"
                last_done = 0

                def _callback(done: int, total: int) -> None:
                    nonlocal last_done
                    if done == 0:
                        frames_progress.total = (frames_progress.total or 0) + total
                        frames_progress.refresh()
                        last_done = 0
                        return
                    delta = max(0, done - last_done)
                    if delta:
                        frames_progress.update(delta)
                    last_done = done

                enhance_file(
                    input_path=wav_path,
                    output_path=out_path,
                    model=args.model,
                    verbose=args.verbose,
                    progress_callback=_callback,
                )
                files_progress.update(1)
                files_progress.set_postfix_str(wav_path.name)
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
