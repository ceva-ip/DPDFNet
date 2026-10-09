#!/usr/bin/env python3
"""Verify a release against checksummed 48 kHz regression fixtures.

Audio is deliberately not distributed with the runtime. --data names the local
fixture root; validation/reference_streams.json records its expected layout.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from dpdfnet import Model
from enhance_wav import spectra, window


def digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def verify_case(job):
    size, case, data, library = job
    path = Path(data) / case["input"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != case["input_sha256"]:
        raise ValueError("Input checksum mismatch: " + case["id"])
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != 48000 or audio.shape != (case["samples"], 1):
        raise ValueError("Fixture shape mismatch")
    expected = case["models"][str(size)]
    output_hash, state_hash = hashlib.sha256(), hashlib.sha256()
    overlap = np.zeros(960, dtype=np.float32)
    win = window()
    chunks = []
    weights = ROOT / "models" / ("dpdfnet%d_48khz_hr" % size) / "weights.f32"
    with Model(library, weights, size) as model:
        prefix = hashlib.sha256()
        for index, frame in enumerate(spectra(audio[:, 0])):
            output = model.process(frame)
            if not np.isfinite(output).all() or not np.isfinite(model.state).all():
                raise AssertionError("Nonfinite model output or state")
            output_hash.update(memoryview(output).cast("B"))
            state_hash.update(memoryview(model.state).cast("B"))
            if index < 8:
                prefix.update(memoryview(output).cast("B")); prefix.update(memoryview(model.state).cast("B"))
            time_frame = np.fft.irfft(output[:, 0] + 1j * output[:, 1], n=960).astype(np.float32) * win
            overlap[:480] = overlap[480:]; overlap[480:] = 0; overlap += time_frame
            chunks.append(overlap[:480].copy())
        actual = {"spectrum_sha256": output_hash.hexdigest(), "state_sha256": state_hash.hexdigest(),
                  "pcm_sha256": digest(np.concatenate(chunks)[2400:2400 + audio.shape[0]])}
        if actual != expected or index + 1 != case["hops"]:
            raise AssertionError("Output/state/PCM regression: model %d, %s" % (size, case["id"]))
        model.reset(); replay = hashlib.sha256()
        for index, frame in enumerate(spectra(audio[:, 0])):
            output = model.process(frame)
            replay.update(memoryview(output).cast("B")); replay.update(memoryview(model.state).cast("B"))
            if index == 7: break
        if replay.digest() != prefix.digest():
            raise AssertionError("State reset regression")
    return {"model": size, "id": case["id"], "hops": case["hops"], "exact": True}


def verify_concurrency(library, size):
    weights = ROOT / "models" / ("dpdfnet%d_48khz_hr" % size) / "weights.f32"
    def stream(index):
        rng = np.random.default_rng(index)
        result = hashlib.sha256()
        with Model(library, weights, size) as model:
            for hop in range(64):
                frame = rng.standard_normal((481, 2)).astype(np.float32) * np.float32((index + 1) / 4)
                frame[0, 1] = frame[-1, 1] = 0
                result.update(memoryview(model.process(frame)).cast("B"))
                result.update(memoryview(model.state).cast("B"))
        return result.hexdigest()
    serial = [stream(i) for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        if list(pool.map(stream, range(4))) != serial:
            raise AssertionError("Independent contexts differ under concurrent calls")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--library", type=Path, default=ROOT / "build/libdpdfnet.so")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output", type=Path, default=ROOT / "validation/local/audio.json")
    args = p.parse_args()
    if args.workers < 1: p.error("--workers must be positive")
    reference = json.loads((ROOT / "validation/reference_streams.json").read_text())
    start = time.monotonic()
    jobs = [(size, case, str(args.data.resolve()), str(args.library.resolve()))
            for size in (2, 8) for case in reference["cases"]]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(verify_case, jobs):
            results.append(result)
            print("Verified model %d: %s" % (result["model"], result["id"]), flush=True)
    for size in (2, 8): verify_concurrency(args.library, size)
    report = {"schema_version": 1, "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "reference_sha256": hashlib.sha256((ROOT / "validation/reference_streams.json").read_bytes()).hexdigest(),
              "models": [2, 8], "cases_per_model": len(reference["cases"]),
              "hops_per_model": sum(case["hops"] for case in reference["cases"]),
              "every_spectrum_state_and_pcm_hash_matches": True, "reset_replay_matches": True,
              "independent_concurrent_contexts_match_serial": True,
              "elapsed_seconds": time.monotonic() - start}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
