#!/usr/bin/env python3
"""Run a mono 48 kHz WAV with a separately installed XDNA2 PCM provider."""

import argparse
import ctypes
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import wave

RATE = 48000
HOP = 480
DELAY = 5 * HOP
DRAIN_HOPS = 6
MAX_HOPS = 1048576


def wav_info(source):
    if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, RATE):
        raise ValueError("Expected mono 16-bit PCM WAV at 48 kHz")
    samples = source.getnframes()
    hops = (samples + HOP - 1) // HOP + DRAIN_HOPS
    if samples == 0 or hops > MAX_HOPS:
        raise ValueError("WAV length is outside the provider's supported range")
    return samples, hops


def provider_assets(catalog_path, model):
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("version") != 1:
        raise ValueError("Expected a version-1 provider catalog")
    try:
        entry = catalog["pcm_profiles"][str(model)]
        verified = []
        for key in ("library", "manifest", "model"):
            path = (catalog_path.parent / entry[key]).resolve(strict=True)
            with path.open("rb") as stream:
                actual = file_digest(stream, "sha256").hexdigest()
            if actual != entry[f"{key}_sha256"]:
                raise ValueError(f"Model {model} provider {key} has the wrong SHA-256")
            verified.append(path)
    except KeyError as error:
        raise ValueError(f"Model {model} needs a complete PCM provider entry") from error
    return verified


def bind(library, name, result, *args):
    function = getattr(library, name)
    function.restype = result
    function.argtypes = args
    return function


def enhance_in_worker(catalog_path, model, source, target):
    import numpy as np

    path, manifest, onnx = provider_assets(catalog_path, model)
    incoming = np.empty(HOP, dtype=np.float32)
    outgoing = np.empty_like(incoming)

    receipt_dir = target.parent / "receipts"
    receipt_dir.mkdir(mode=0o700)
    os.environ["DPDF_R103_OUTPUT_DIRECTORY"] = str(receipt_dir)
    library = ctypes.CDLL(str(path))
    abi = bind(library, "r103_abi_version_v1", ctypes.c_uint32)
    validate = bind(library, "r103_config_only_v1", ctypes.c_int, ctypes.c_char_p, ctypes.c_int)
    create = bind(library, "audio_native_create", ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p)
    step = bind(library, "audio_native_step", ctypes.c_int,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    close = bind(library, "r103_destroy_quiet_v1", ctypes.c_int, ctypes.c_void_p)
    if abi() != 1 or validate(os.fsencode(manifest), model) != 0:
        raise RuntimeError("XDNA2 provider ABI or model configuration is incompatible")
    handle = create(os.fsencode(manifest), os.fsencode(onnx))
    if not handle:
        raise RuntimeError("XDNA2 could not open the model")
    try:
        input_address, output_address = incoming.ctypes.data, outgoing.ctypes.data
        with wave.open(str(source), "rb") as wav, target.open("xb") as stream, wave.open(stream, "wb") as result:
            samples, hops = wav_info(wav)
            result.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            for offset in range(0, hops * HOP, HOP):
                count = min(HOP, max(0, samples - offset))
                original = np.frombuffer(wav.readframes(count), dtype="<i2")
                if original.size != count:
                    raise ValueError("Truncated WAV data")
                incoming[:count] = original * (1 / 32768)
                incoming[count:] = 0
                if step(handle, input_address, output_address) != 0:
                    raise RuntimeError("XDNA2 failed to process an audio frame")
                if not np.isfinite(outgoing).all():
                    raise RuntimeError("XDNA2 returned non-finite audio")
                if DELAY <= offset < samples + DELAY:
                    count = min(HOP, samples + DELAY - offset)
                    pcm = np.rint(outgoing[:count] * 32768).clip(-32768, 32767).astype("<i2")
                    result.writeframesraw(pcm.tobytes())
    finally:
        close_status = close(handle)
    if close_status != 0:
        raise RuntimeError("XDNA2 provider shutdown failed")


def run_preview(catalog, model, source, target):
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    with wave.open(str(source), "rb") as wav:
        _, hops = wav_info(wav)
    with TemporaryDirectory(prefix=".dpdfnet-xdna2-", dir=target.parent) as folder:
        temporary = Path(folder) / "output.wav"
        subprocess.run([sys.executable, "-P", str(Path(__file__).resolve()),
                        "--providers", str(catalog), "--model", str(model), "--worker",
                        str(source), str(temporary)],
                       check=True, timeout=30 + hops * 0.03)
        os.link(temporary, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--providers", type=Path, required=True)
    parser.add_argument("--model", type=int, choices=(2, 8), default=8)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        action = enhance_in_worker if args.worker else run_preview
        action(args.providers, args.model, args.input, args.output)
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError, wave.Error,
            subprocess.SubprocessError) as error:
        parser.exit(1, f"XDNA2 preview: {error}\n")


if __name__ == "__main__":
    main()
