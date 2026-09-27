"""Publisher check: packaged C ABI matches the measured research builds exactly.

Run from native_inference; consumers do not need Python or the research files.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extended_probe import ExtendedModel
from probe import FP, ptr, session, initial_state, spectra


class API(ct.Structure):
    _fields_ = [('abi_version', ct.c_uint32), ('struct_size', ct.c_uint32),
                ('model_name', ct.c_char_p), ('weights_sha256', ct.c_char_p),
                ('sample_rate', ct.c_uint32), ('hop_size', ct.c_uint32),
                ('spectrum_size', ct.c_size_t), ('state_size', ct.c_size_t),
                ('weight_count', ct.c_size_t),
                ('supported', ct.CFUNCTYPE(ct.c_int, ct.c_uint32)),
                ('create', ct.CFUNCTYPE(ct.c_void_p, FP, ct.c_size_t, ct.c_uint32)),
                ('init', ct.CFUNCTYPE(ct.c_int, FP)),
                ('process', ct.CFUNCTYPE(ct.c_int, ct.c_void_p, FP, FP, FP, FP)),
                ('destroy', ct.CFUNCTYPE(None, ct.c_void_p)),
                ('owned', ct.CFUNCTYPE(ct.c_size_t, ct.c_void_p))]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--w7', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    library = args.build/'libdpdf_native.so'
    lib = ct.CDLL(str(library.resolve()))
    report = {'library_sha256': sha(library), 'w7': args.w7, 'models': {}}
    for size in (2, 8):
        name = f'dpdfnet{size}_48khz_hr'
        getter = getattr(lib, name+'_get_api')
        getter.argtypes = [ct.c_uint32]
        getter.restype = ct.POINTER(API)
        api = getter(1).contents
        assert api.abi_version == 1 and api.struct_size == ct.sizeof(API)
        weights = Path(f'artifacts/v1/{name}/weights.f32')
        assert sha(weights) == api.weights_sha256.decode()
        w = np.fromfile(weights, dtype='<f4')
        preset = 2 if args.w7 else 1
        assert api.supported(preset) and not api.supported(3-preset)
        handle = api.create(ptr(w), w.size, preset)
        assert handle
        reference = session(f'models/{name}.onnx')
        research = Path('build/w7_followup_pack_fit5' + ('_2' if size == 2 else '')) if args.w7 else Path(f'build/followup_final{size}')
        candidate = ExtendedModel(reference, 4, 8, 7, build=research, weights=weights)
        state = np.empty(api.state_size, dtype=np.float32)
        assert api.init(ptr(state)) == 0
        other = initial_state(reference)
        assert np.array_equal(state, other)
        y = np.empty((1, 1, 481, 2), dtype=np.float32)
        for x in spectra(300):
            assert api.process(handle, ptr(x), ptr(state), ptr(y), ptr(state)) == 0
            expected, other = candidate.run(None, {'spec': x, 'state_in': other})
            assert np.array_equal(y, expected) and np.array_equal(state, other)
        assert api.owned(handle) == candidate.owned_bytes
        report['models'][name] = {'frames': 300, 'output_and_state_bit_exact': True,
            'owned_bytes': api.owned(handle), 'research_sha256': sha(research/'libdpdf_full.so')}
        candidate.close()
        api.destroy(handle)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
