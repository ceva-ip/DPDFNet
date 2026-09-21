"""Fresh-process, warmed model RSS above a common imported Python/runtime baseline.

Run once per model/configuration. Both paths import the same dependencies before
sampling. No reference ONNX session is created in a native measurement. Native
source weights are memory-mapped and unmapped before the final sample.
"""
import argparse
import ctypes as ct
import gc
import json
import mmap
from pathlib import Path

import numpy as np
from probe import FP, ptr, session, initial_state
from model_precision_final import FINAL_CONFIGS


def memory_status():
    fields = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('VmRSS', 'VmHWM'):
            fields[key] = int(value.split()[0]) * 1024
    return fields


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--variant', choices=['original_fp32', *FINAL_CONFIGS], required=True)
    a = p.parse_args()
    gc.collect()
    before = memory_status()
    owned = None
    if a.variant == 'original_fp32':
        model = session(a.model)
        state = initial_state(model)
        x = np.zeros((1, 1, 481, 2), dtype=np.float32)
        for i in range(120):
            x.flat[i % x.size] = .01
            y, state = model.run(None, {'spec': x, 'state_in': state})
        assert np.isfinite(y).all() and np.isfinite(state).all()
    else:
        lib = ct.CDLL(str((a.build / 'libdpdf_full.so').resolve()))
        lib.dpdf_model_create_config.argtypes = [FP, ct.c_size_t, ct.c_int, ct.c_int, ct.c_uint]
        lib.dpdf_model_create_config.restype = ct.c_void_p
        lib.dpdf_model_state_size.restype = ct.c_size_t
        lib.dpdf_model_init_state.argtypes = [FP]
        lib.dpdf_model_process.argtypes = [ct.c_void_p, FP, FP, FP, FP]
        lib.dpdf_model_owned_bytes.argtypes = [ct.c_void_p]
        lib.dpdf_model_owned_bytes.restype = ct.c_size_t
        lib.dpdf_model_destroy.argtypes = [ct.c_void_p]
        with a.weights.open('rb') as source:
            with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
                weights = np.frombuffer(mapping, dtype='<f4')
                handle = lib.dpdf_model_create_config(ptr(weights), weights.size, *FINAL_CONFIGS[a.variant])
                del weights
        if not handle:
            raise RuntimeError('Native model creation failed')
        owned = int(lib.dpdf_model_owned_bytes(handle))
        state = np.zeros(lib.dpdf_model_state_size(), dtype=np.float32)
        assert lib.dpdf_model_init_state(ptr(state)) == 0
        x = np.zeros((1, 1, 481, 2), dtype=np.float32)
        y = np.empty_like(x)
        for i in range(120):
            x.flat[i % x.size] = .01
            next_state = np.empty_like(state)
            assert lib.dpdf_model_process(handle, ptr(x), ptr(state), ptr(y), ptr(next_state)) == 0
            state = next_state
        assert np.isfinite(y).all() and np.isfinite(state).all()
    after = memory_status()
    print(json.dumps({'rss_before_bytes': before['VmRSS'], 'rss_after_bytes': after['VmRSS'],
                      'rss_delta_bytes': after['VmRSS'] - before['VmRSS'],
                      'peak_rss_bytes': after['VmHWM'], 'owned_bytes': owned}))
    if a.variant != 'original_fp32':
        lib.dpdf_model_destroy(handle)


if __name__ == '__main__':
    main()
