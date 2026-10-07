"""Byte-exact block/layout/alias oracle for isolated DPRNN copy experiments.

Runs correctness calls only; no timings are collected. The two libraries must
implement the same baseline arithmetic. Both actual block fixtures and a
signed-zero synthetic weight fixture are covered.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import numpy as np


FLOAT_PTR = ct.POINTER(ct.c_float)
WEIGHT_COUNT = 87552


def pointer(array):
    return array.ctypes.data_as(FLOAT_PTR)


def load(path):
    lib = ct.CDLL(str(path.resolve()))
    lib.dpdf_create.argtypes = [ct.c_int, FLOAT_PTR, ct.c_size_t,
                               ct.c_float, ct.c_float, ct.c_int]
    lib.dpdf_create.restype = ct.c_void_p
    for name in ("dpdf_create_fp16", "dpdf_create_int8"):
        func = getattr(lib, name)
        func.argtypes = [ct.c_int, FLOAT_PTR, ct.c_size_t, ct.c_float, ct.c_float]
        func.restype = ct.c_void_p
    lib.dpdf_destroy.argtypes = [ct.c_void_p]
    lib.dpdf_process_layout.argtypes = [ct.c_void_p, FLOAT_PTR, FLOAT_PTR,
                                       FLOAT_PTR, FLOAT_PTR, ct.c_uint]
    lib.dpdf_process_layout.restype = ct.c_int
    return lib


def create(lib, weights, freq, mode):
    args = [freq, pointer(weights), WEIGHT_COUNT, 1e-5, 1e-5]
    if mode == "scalar":
        result = lib.dpdf_create(*args, 1)
    elif mode == "fp32":
        result = lib.dpdf_create(*args, 2)
    elif mode == "fp16":
        result = lib.dpdf_create_fp16(*args)
    else:
        result = lib.dpdf_create_int8(*args)
    if not result:
        raise RuntimeError(f"Block creation failed: {mode}, F={freq}")
    return result


def raw_equal(a, b, label):
    equal = np.array_equal(a.view(np.uint32), b.view(np.uint32))
    if not equal:
        indexes = np.flatnonzero(a.ravel().view(np.uint32) != b.ravel().view(np.uint32))
        index = int(indexes[0])
        raise AssertionError(f"{label}: first unequal float={index}, "
                             f"bits={a.ravel().view(np.uint32)[index]:08x}/"
                             f"{b.ravel().view(np.uint32)[index]:08x}")


def call(lib, block, features, old, flags, alias_features, alias_state):
    # Features here are canonical [F,64]. Test all input/output layout pairs,
    # including in-place layout conversion, with unaligned valid float buffers.
    packed = features if flags & 1 else features.T
    storage = np.zeros(features.size + 2, dtype=np.float32)
    source = storage[1:-1]
    source[:] = packed.ravel()
    state_storage = np.zeros(old.size + 2, dtype=np.float32)
    state = state_storage[1:-1]
    state[:] = old.ravel()
    output_storage = np.full(features.size + 2, np.float32(12345), dtype=np.float32)
    out = source if alias_features else output_storage[1:-1]
    next_storage = np.full(old.size + 2, np.float32(-12345), dtype=np.float32)
    next_state = state if alias_state else next_storage[1:-1]
    if lib.dpdf_process_layout(block, pointer(source), pointer(state), pointer(out),
                               pointer(next_state), flags):
        raise AssertionError("process_layout returned an error")
    if storage[0] != 0 or storage[-1] != 0 or state_storage[0] != 0 or state_storage[-1] != 0:
        raise AssertionError("Input/state guard overwritten")
    if output_storage[0] != 12345 or output_storage[-1] != 12345:
        raise AssertionError("Output guard overwritten")
    if next_storage[0] != -12345 or next_storage[-1] != -12345:
        raise AssertionError("Next-state guard overwritten")
    canonical = out.reshape(features.shape) if flags & 2 else out.reshape(64, -1).T
    return canonical.copy(), next_state.reshape(old.shape).copy()


def patterns(freq):
    rng = np.random.default_rng(20261007 + freq)
    values = []
    signed_zero = np.zeros((freq, 64), dtype=np.float32)
    signed_zero.view(np.uint32).ravel()[::2] = 0x80000000
    values.append(("alternating_signed_zero", signed_zero))
    for label, scale in (("quiet", 1e-5), ("normal", 1), ("high", 100)):
        array = (rng.normal(size=(freq, 64))*scale).astype(np.float32)
        array.ravel()[::17] = 0
        array.ravel().view(np.uint32)[::31] = 0x80000000
        values.append((label, array))
    return values


def run(args):
    libraries = [load(path) for path in (args.baseline, args.candidate)]
    fixtures = []
    for path in args.weights:
        weights = np.fromfile(path, dtype=np.float32)
        if weights.size != WEIGHT_COUNT:
            raise ValueError(f"Expected {WEIGHT_COUNT} floats: {path}")
        fixtures.append((str(path), weights))
    zeros = np.zeros(WEIGHT_COUNT, dtype=np.float32)
    # Deliberately include negative zeros in matrices, gate/projection bias,
    # norm scales/bias and retain otherwise valid positive epsilons.
    zeros.view(np.uint32)[::2] = 0x80000000
    fixtures.append(("synthetic_signed_zero_weights", zeros))
    checks = 0
    details = []
    for fixture, weights in fixtures:
        for freq in (40, 48):
            cases = patterns(freq)
            for mode in args.modes:
                blocks = [create(lib, weights, freq, mode) for lib in libraries]
                try:
                    for feature_name, features in cases:
                        for state_name, state in cases:
                            expected = call(libraries[0], blocks[0], features, state, 0, False, False)
                            for flags in range(4):
                                for alias_features in (False, True):
                                    for alias_state in (False, True):
                                        actual = call(libraries[1], blocks[1], features, state,
                                                      flags, alias_features, alias_state)
                                        label = (f"{fixture}/{freq}/{mode}/{feature_name}/{state_name}/"
                                                 f"flags={flags}/alias={alias_features},{alias_state}")
                                        for part, a, b in zip(("features", "state"), expected, actual):
                                            raw_equal(a, b, label + "/" + part)
                                        checks += 1
                finally:
                    for lib, block in zip(libraries, blocks):
                        lib.dpdf_destroy(block)
                details.append({"fixture": fixture, "frequency_rows": freq,
                                "mode": mode, "cases": len(cases)**2*16, "exact": True})
                print(f"{fixture}: F={freq}, {mode}: exact layouts, aliases, signed zeros", flush=True)
    report = {"passed": True, "byte_exact_block_calls": checks,
              "layout_flags": [0, 1, 2, 3], "unaligned_buffers": True,
              "input_output_and_state_aliasing": True, "signed_zero_bit_checks": True,
              "libraries": {name: {"path": str(path),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                            for name, path in (("baseline", args.baseline), ("candidate", args.candidate))},
              "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--modes", choices=("scalar", "fp32", "fp16", "w7a8"),
                        nargs="+", default=["scalar", "fp32", "fp16", "w7a8"])
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
