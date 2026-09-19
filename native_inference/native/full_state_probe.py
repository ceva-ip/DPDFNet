"""Ensure independent model arenas/states reproduce serial results concurrently."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
from full_probe import FullModel
from probe import session, initial_state, spectra


def main():
    ref = session('models/dpdfnet8_48khz_hr.onnx')
    frames = spectra(64)
    results = {}
    for tier, name in [(0, 'fp32'), (3, 'fp16'), (4, 'int8')]:
        def run(stream):
            model = FullModel(Path('build/full/libdpdf_full.so'), Path('models/full_c/weights.f32'), ref, tier)
            try:
                state = initial_state(ref); out = []
                for x in frames:
                    y, state = model.run(None, {'spec': np.ascontiguousarray(x*(stream+1)/4), 'state_in': state})
                    out.append(y)
                return np.stack(out), state
            finally:
                model.close()
        serial = [run(i) for i in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            concurrent = list(pool.map(run, range(4)))
        assert all(np.array_equal(a, b) for left, right in zip(serial, concurrent) for a, b in zip(left, right))
        results[name] = {'streams': 4, 'frames_per_stream': 64, 'bit_identical_to_serial': True}
        print(name, 'independent state/arena concurrency passed', flush=True)
    Path('results/full_state_validation.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    main()
