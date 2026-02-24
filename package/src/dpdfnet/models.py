from __future__ import annotations

from dataclasses import asdict, dataclass
import errno
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from filelock import FileLock


@dataclass(frozen=True)
class ModelInfo:
    name: str
    sample_rate: int
    frame_ms: float
    description: str
    onnx_filename: str
    state_filename: str


MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "baseline": ModelInfo(
        name="baseline",
        sample_rate=16000,
        frame_ms=20.0,
        description="Fastest and lowest-compute baseline model.",
        onnx_filename="baseline.onnx",
        state_filename="baseline_state.npz",
    ),
    "dpdfnet2": ModelInfo(
        name="dpdfnet2",
        sample_rate=16000,
        frame_ms=20.0,
        description="Balanced quality/speed DPDFNet-2 model.",
        onnx_filename="dpdfnet2.onnx",
        state_filename="dpdfnet2_state.npz",
    ),
    "dpdfnet4": ModelInfo(
        name="dpdfnet4",
        sample_rate=16000,
        frame_ms=20.0,
        description="Higher quality DPDFNet-4 model.",
        onnx_filename="dpdfnet4.onnx",
        state_filename="dpdfnet4_state.npz",
    ),
    "dpdfnet8": ModelInfo(
        name="dpdfnet8",
        sample_rate=16000,
        frame_ms=20.0,
        description="Highest quality 16 kHz DPDFNet-8 model.",
        onnx_filename="dpdfnet8.onnx",
        state_filename="dpdfnet8_state.npz",
    ),
    "dpdfnet2_48khz_hr": ModelInfo(
        name="dpdfnet2_48khz_hr",
        sample_rate=48000,
        frame_ms=20.0,
        description="High-resolution 48 kHz DPDFNet-2 model.",
        onnx_filename="dpdfnet2_48khz_hr.onnx",
        state_filename="dpdfnet2_48khz_hr_state.npz",
    ),
}

DEFAULT_MODEL = "dpdfnet2"
DEFAULT_REVISION = "main"
DEFAULT_HF_REPO = "Ceva-IP/DPDFNet"
DEFAULT_HF_BASE = "https://huggingface.co"
DEFAULT_HF_SUBDIR = "onnx"
DEFAULT_DOWNLOAD_RETRIES = 3


@dataclass(frozen=True)
class ResolvedModel:
    info: ModelInfo
    onnx_path: Path
    state_path: Path


def _unique_paths(paths: List[Path]) -> List[Path]:
    unique: List[Path] = []
    seen: set[str] = set()
    for item in paths:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _default_cache_root() -> Path:
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / "dpdfnet"
        return Path.home() / "AppData" / "Local" / "dpdfnet"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "dpdfnet"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "dpdfnet"
    return Path.home() / ".cache" / "dpdfnet"


def get_cache_dir() -> Path:
    override = os.environ.get("DPDFNET_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return _default_cache_root().resolve()


def get_cache_model_dir() -> Path:
    return get_cache_dir() / "models"


def _download_target_dir() -> Path:
    env_model_dir = os.environ.get("DPDFNET_MODEL_DIR")
    if env_model_dir:
        return Path(env_model_dir).expanduser().resolve()
    return get_cache_model_dir().resolve()


def _candidate_model_dirs() -> List[Path]:
    env_dir = os.environ.get("DPDFNET_MODEL_DIR")
    if env_dir:
        # Explicit model directory overrides cache search to keep resolution predictable.
        return [Path(env_dir).expanduser().resolve()]

    return [get_cache_model_dir().resolve()]


def _is_valid_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _emit(message: str, notifier: Optional[Callable[[str], None]]) -> None:
    if notifier is not None:
        notifier(message)
        return
    print(message, file=sys.stderr)


def _hf_url(filename: str, revision: str) -> str:
    repo = os.environ.get("DPDFNET_HF_REPO", DEFAULT_HF_REPO).strip("/")
    base = os.environ.get("DPDFNET_HF_BASE_URL", DEFAULT_HF_BASE).rstrip("/")
    subdir = os.environ.get("DPDFNET_HF_SUBDIR", DEFAULT_HF_SUBDIR).strip("/")
    remote_path = f"{subdir}/{filename}" if subdir else filename
    return f"{base}/{repo}/resolve/{revision}/{remote_path}?download=true"


def _download_one(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.part.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(temp_fd, "wb") as out:
            with urlopen(url, timeout=60) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp_name, destination)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _assert_writable_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to create model directory: {path}. "
            f"Set DPDFNET_CACHE_DIR or DPDFNET_MODEL_DIR to a writable location. ({exc})"
        ) from exc

    probe_fd: Optional[int] = None
    probe_name: Optional[str] = None
    try:
        probe_fd, probe_name = tempfile.mkstemp(prefix=".dpdfnet.write-test.", dir=str(path))
    except OSError as exc:
        raise RuntimeError(
            f"Model directory is not writable: {path}. "
            f"Set DPDFNET_CACHE_DIR or DPDFNET_MODEL_DIR to a writable location. ({exc})"
        ) from exc
    finally:
        if probe_fd is not None:
            os.close(probe_fd)
        if probe_name is not None:
            try:
                os.unlink(probe_name)
            except OSError:
                pass


def get_model_info(model: str) -> ModelInfo:
    try:
        return MODEL_REGISTRY[model]
    except KeyError as exc:
        supported = ", ".join(supported_models())
        raise ValueError(f"Unsupported model '{model}'. Supported: {supported}") from exc


def supported_models() -> List[str]:
    return sorted(MODEL_REGISTRY)


def _download_retries() -> int:
    raw = os.environ.get("DPDFNET_DOWNLOAD_RETRIES", str(DEFAULT_DOWNLOAD_RETRIES))
    try:
        retries = int(raw)
    except ValueError:
        retries = DEFAULT_DOWNLOAD_RETRIES
    return max(1, retries)


def _is_retryable_http_error(exc: HTTPError) -> bool:
    return exc.code in {408, 425, 429, 500, 502, 503, 504}


def _is_retryable_os_error(exc: OSError) -> bool:
    return exc.errno in {
        errno.ECONNABORTED,
        errno.ECONNRESET,
        errno.ETIMEDOUT,
        errno.ENETRESET,
        errno.ENETUNREACH,
        errno.EHOSTUNREACH,
    }


def _download_with_retries(
    *,
    url: str,
    destination: Path,
    verbose: bool,
    notifier: Optional[Callable[[str], None]],
) -> None:
    attempts = _download_retries()
    for attempt in range(1, attempts + 1):
        try:
            _download_one(url, destination)
            return
        except HTTPError as exc:
            if not _is_retryable_http_error(exc) or attempt >= attempts:
                raise
            wait_s = min(8.0, 0.5 * (2 ** (attempt - 1)))
            if verbose:
                _emit(
                    f"  transient HTTP {exc.code} downloading {destination.name}; retrying in {wait_s:.1f}s "
                    f"({attempt}/{attempts})",
                    notifier,
                )
            time.sleep(wait_s)
        except URLError as exc:
            if attempt >= attempts:
                raise
            wait_s = min(8.0, 0.5 * (2 ** (attempt - 1)))
            if verbose:
                _emit(
                    f"  transient network error downloading {destination.name}; retrying in {wait_s:.1f}s "
                    f"({attempt}/{attempts})",
                    notifier,
                )
            time.sleep(wait_s)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EPERM, errno.EROFS}:
                raise
            if not _is_retryable_os_error(exc) or attempt >= attempts:
                raise
            wait_s = min(8.0, 0.5 * (2 ** (attempt - 1)))
            if verbose:
                _emit(
                    f"  transient I/O error downloading {destination.name}; retrying in {wait_s:.1f}s "
                    f"({attempt}/{attempts})",
                    notifier,
                )
            time.sleep(wait_s)


def _ensure_downloaded(
    *,
    info: ModelInfo,
    destination_dir: Path,
    revision: str,
    force: bool,
    verbose: bool,
    notifier: Optional[Callable[[str], None]],
) -> None:
    destination_dir = destination_dir.expanduser().resolve()
    _assert_writable_dir(destination_dir)
    onnx_path = destination_dir / info.onnx_filename
    state_path = destination_dir / info.state_filename

    lock = FileLock(str(destination_dir / f".{info.name}.download.lock"))
    with lock:
        if not force and _is_valid_file(onnx_path) and _is_valid_file(state_path):
            return

        action = "Refreshing" if force else "Downloading"
        _emit(f"{action} model '{info.name}' to {destination_dir}", notifier)
        assets = [
            (onnx_path, info.onnx_filename),
            (state_path, info.state_filename),
        ]
        for file_path, filename in assets:
            if not force and _is_valid_file(file_path):
                continue
            url = _hf_url(filename, revision)
            if verbose:
                _emit(f"  {filename} <- {url}", notifier)
            try:
                _download_with_retries(
                    url=url,
                    destination=file_path,
                    verbose=verbose,
                    notifier=notifier,
                )
            except HTTPError as exc:
                detail = f"HTTP {exc.code}" + (f" ({exc.reason})" if exc.reason else "")
                raise RuntimeError(
                    f"Failed to download '{filename}' from '{url}'. "
                    f"{detail}. Confirm access to Hugging Face and retry. "
                    f"You can also pre-download using: dpdfnet download {info.name}"
                ) from exc
            except URLError as exc:
                raise RuntimeError(
                    f"Failed to download '{filename}' from '{url}'. "
                    f"Network error: {exc.reason}. Check network/proxy settings and retry. "
                    f"You can also pre-download using: dpdfnet download {info.name}"
                ) from exc
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EPERM, errno.EROFS}:
                    raise RuntimeError(
                        f"Failed to write '{filename}' to '{destination_dir}'. "
                        f"Set DPDFNET_CACHE_DIR or DPDFNET_MODEL_DIR to a writable location. ({exc})"
                    ) from exc
                raise RuntimeError(
                    f"Failed to download '{filename}' from '{url}'. "
                    f"Local filesystem error while writing '{file_path}': {exc}. "
                    f"You can also pre-download using: dpdfnet download {info.name}"
                ) from exc

        if not (_is_valid_file(onnx_path) and _is_valid_file(state_path)):
            raise RuntimeError(
                f"Downloaded files for model '{info.name}' are invalid in {destination_dir}. "
                "Please retry after removing the files."
            )


def _find_first_existing(paths: List[Path], filename: str) -> Optional[Path]:
    for directory in paths:
        candidate = directory / filename
        if _is_valid_file(candidate):
            return candidate.resolve()
    return None


def resolve_model(
    *,
    model: str,
    onnx_path: Optional[Union[str, Path]] = None,
    state_path: Optional[Union[str, Path]] = None,
    auto_download: bool = True,
    verbose: bool = False,
    notifier: Optional[Callable[[str], None]] = None,
) -> ResolvedModel:
    info = get_model_info(model)
    search_dirs = _candidate_model_dirs()
    chosen_onnx: Optional[Path] = None

    if onnx_path is not None:
        explicit_onnx = Path(onnx_path).expanduser().resolve()
        if not _is_valid_file(explicit_onnx):
            raise FileNotFoundError(f"ONNX model file not found or empty: {explicit_onnx}")
        chosen_onnx = explicit_onnx
    else:
        chosen_onnx = _find_first_existing(search_dirs, info.onnx_filename)
        if chosen_onnx is None and auto_download:
            target = _download_target_dir()
            _ensure_downloaded(
                info=info,
                destination_dir=target,
                revision=DEFAULT_REVISION,
                force=False,
                verbose=verbose,
                notifier=notifier,
            )
            chosen_onnx = (target / info.onnx_filename).resolve()

    if chosen_onnx is None or not _is_valid_file(chosen_onnx):
        searched = [str(p) for p in search_dirs]
        raise FileNotFoundError(
            f"Could not resolve ONNX model for '{info.name}'. Searched: {searched}. "
            "Set DPDFNET_CACHE_DIR/DPDFNET_MODEL_DIR, or use Python API onnx_path/state_path."
        )

    if state_path is not None:
        chosen_state = Path(state_path).expanduser().resolve()
    elif onnx_path is not None:
        chosen_state = chosen_onnx.with_name(f"{chosen_onnx.stem}_state.npz")
    else:
        chosen_state = chosen_onnx.with_name(info.state_filename)

    if not _is_valid_file(chosen_state):
        if state_path is None and onnx_path is None:
            fallback_state = _find_first_existing(search_dirs, info.state_filename)
            if fallback_state is not None:
                chosen_state = fallback_state
        if not _is_valid_file(chosen_state) and state_path is None and onnx_path is None and auto_download:
            target = _download_target_dir()
            _ensure_downloaded(
                info=info,
                destination_dir=target,
                revision=DEFAULT_REVISION,
                force=False,
                verbose=verbose,
                notifier=notifier,
            )
            chosen_onnx = (target / info.onnx_filename).resolve()
            chosen_state = (target / info.state_filename).resolve()

    if not _is_valid_file(chosen_state):
        raise FileNotFoundError(
            f"State file not found or empty: {chosen_state}. "
            "State files must be named <model>_state.npz unless explicitly provided."
        )

    return ResolvedModel(info=info, onnx_path=chosen_onnx, state_path=chosen_state)


def download_model(
    *,
    model: str,
    force: bool = False,
    verbose: bool = False,
    notifier: Optional[Callable[[str], None]] = None,
) -> ResolvedModel:
    info = get_model_info(model)
    target = _download_target_dir()
    _ensure_downloaded(
        info=info,
        destination_dir=target,
        revision=DEFAULT_REVISION,
        force=force,
        verbose=verbose,
        notifier=notifier,
    )
    return ResolvedModel(
        info=info,
        onnx_path=(target / info.onnx_filename).resolve(),
        state_path=(target / info.state_filename).resolve(),
    )


def download_models(
    *,
    models: Optional[List[str]] = None,
    force: bool = False,
    verbose: bool = False,
    notifier: Optional[Callable[[str], None]] = None,
) -> List[ResolvedModel]:
    names = supported_models() if models is None else [get_model_info(model).name for model in models]
    results: List[ResolvedModel] = []
    for model in names:
        results.append(
            download_model(
                model=model,
                force=force,
                verbose=verbose,
                notifier=notifier,
            )
        )
    return results


def available_model_entries() -> List[Dict[str, Any]]:
    search_dirs = _candidate_model_dirs()
    cache_dir = get_cache_model_dir().resolve()
    entries: List[Dict[str, Any]] = []
    for name in supported_models():
        info = MODEL_REGISTRY[name]
        onnx_path = _find_first_existing(search_dirs, info.onnx_filename)
        state_path = _find_first_existing(search_dirs, info.state_filename)
        row = asdict(info)
        row["onnx_path"] = str(onnx_path) if onnx_path else None
        row["state_path"] = str(state_path) if state_path else None
        row["onnx_found"] = onnx_path is not None
        row["state_found"] = state_path is not None
        row["ready"] = onnx_path is not None and state_path is not None
        row["cache_dir"] = str(cache_dir)
        row["cached"] = _is_valid_file(cache_dir / info.onnx_filename) and _is_valid_file(
            cache_dir / info.state_filename
        )
        entries.append(row)

    return entries
