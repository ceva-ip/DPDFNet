"""Parity and like-for-like timings for original, hybrid, and standalone C."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from probe import (FP, ptr, session, initial_state, spectra, whole_parity,
                   audio_spectra, summarize, cpu_name)


class FullModel:
    def __init__(self, path, weights, reference, tier=0):
        self.reference = reference
        self.lib = ct.CDLL(str(path.resolve()))
        lib = self.lib
        lib.dpdf_model_weights_sha256.restype = ct.c_char_p
        lib.dpdf_model_weight_count.restype = ct.c_size_t
        lib.dpdf_model_create.argtypes = [FP, ct.c_size_t, ct.c_int]
        lib.dpdf_model_create.restype = ct.c_void_p
        lib.dpdf_model_destroy.argtypes = [ct.c_void_p]
        lib.dpdf_model_destroy.restype = None
        lib.dpdf_model_process.argtypes = [ct.c_void_p, FP, FP, FP, FP]
        lib.dpdf_model_process.restype = ct.c_int
        lib.dpdf_model_init_state.argtypes = [FP]
        lib.dpdf_model_init_state.restype = ct.c_int
        data = weights.read_bytes()
        assert hashlib.sha256(data).hexdigest() == lib.dpdf_model_weights_sha256().decode()
        w = np.frombuffer(data, dtype='<f4')
        assert w.size == lib.dpdf_model_weight_count()
        self.handle = lib.dpdf_model_create(ptr(w), w.size, tier)
        if not self.handle:
            raise RuntimeError('C model creation failed')
        state = np.empty(initial_state(reference).size, dtype=np.float32)
        assert lib.dpdf_model_init_state(ptr(state)) == 0
        assert np.array_equal(state, initial_state(reference))

    def get_inputs(self):
        return self.reference.get_inputs()

    def get_modelmeta(self):
        return self.reference.get_modelmeta()

    def run(self, unused, inputs):
        # Match ORT's returned-output ownership, including allocations in timing.
        y = np.empty((1, 1, 481, 2), dtype=np.float32)
        state = np.empty_like(inputs['state_in'])
        rc = self.lib.dpdf_model_process(self.handle, ptr(inputs['spec']),
                                        ptr(inputs['state_in']), ptr(y), ptr(state))
        if rc:
            raise RuntimeError(f'C model process failed: {rc}')
        return y, state

    def close(self):
        if self.handle:
            self.lib.dpdf_model_destroy(self.handle)
            self.handle = None


def timings(sessions, frames, repeats, paced, warmup=100):
    result = {k: [] for k in sessions}
    for repeat in range(repeats):
        names = list(sessions)
        names = names[repeat % len(names):] + names[:repeat % len(names)]
        for name in names:
            sess = sessions[name]
            state = initial_state(sess)
            times = []
            start = time.perf_counter()
            for i, x in enumerate(frames):
                if paced:
                    delay = start + i * .01 - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                begin = time.perf_counter_ns()
                _, state = sess.run(None, {'spec': x, 'state_in': state})
                elapsed = (time.perf_counter_ns() - begin) / 1e6
                if i >= warmup:
                    times.append(elapsed)
            result[name].append(summarize(times))
            print(f'{name} paced={paced} repeat={repeat}: {result[name][-1]}', flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', type=int, default=1000)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--build', type=Path, default=Path('build/full'))
    p.add_argument('--audio', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, default=Path('results/full_c.json'))
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--paced', action='store_true')
    a = p.parse_args()
    if a.frames <= 0 or a.repeats <= 0:
        p.error('Positive frame/repeat counts required')
    ref = session('models/dpdfnet8_48khz_hr.onnx')
    full = FullModel(a.build / 'libdpdf_full.so', Path('models/full_c/weights.f32'), ref)
    frames = spectra(a.frames + 100)
    result = {'environment': {'cpu': cpu_name(), 'platform': platform.platform()},
              'config': {'frames': a.frames, 'warmup': 100, 'repeats': a.repeats, 'threads': 1},
              'weights_sha256': full.lib.dpdf_model_weights_sha256().decode(),
              'parity': [whole_parity(ref, full, frames, 'synthetic')]}
    print('Synthetic parity:', result['parity'][-1], flush=True)
    for path in a.audio:
        parity = whole_parity(ref, full, audio_spectra(path), path.name)
        parity['audio_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        result['parity'].append(parity)
        print('Audio parity:', parity, flush=True)
    if not a.check_only:
        hybrid = session('models/native_blocks/hybrid.onnx', 'build/baseline/libdpdf_ort.so')
        sessions = {'onnx': ref, 'native_hybrid': hybrid, 'full_c': full}
        result['timings'] = timings(sessions, frames, a.repeats, False)
        if a.paced:
            result['paced_timings'] = timings(sessions, frames, a.repeats, True)
    full.close()
    a.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
