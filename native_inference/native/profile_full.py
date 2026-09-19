"""Produce/run an instrumented diagnostic copy; never benchmark this build."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import re
import numpy as np
from full_probe import FullModel
from probe import session, spectra, initial_state, ptr, FP


def instrument():
    source = Path('models/full_c/generated_model.c').read_text()
    comments = list(re.finditer(r'/\* (\d+): (\w+) [^\n]*\*/', source))
    start = source.index('int dpdf_model_process(')
    prefix = source[:start]
    prefix = '#include <time.h>\n' + prefix
    prefix += ('static double timings[377];\n'
               'static double stamp(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }\n'
               'void dpdf_profile_get(double *out) { memcpy(out,timings,sizeof(timings)); }\n')
    body = source[start:]
    for match in reversed(comments):
        pos = match.end() - start
        number = int(match[1])
        code = ('double begin=stamp();' if number == 0 else
                f'timings[{number-1}]+=stamp()-begin; begin=stamp();')
        body = body[:pos] + '\n' + code + body[pos:]
    body = body.replace('return 0;\n}', 'timings[376]+=stamp()-begin; return 0;\n}')
    Path('models/full_c/profile_model.c').write_text(prefix + body)
    Path('models/full_c/profile_nodes.json').write_text(json.dumps([m.group(0) for m in comments]))


def run():
    ref = session('models/dpdfnet8_48khz_hr.onnx')
    full = FullModel(Path('build/full-profile/libdpdf_full.so'), Path('models/full_c/weights.f32'), ref)
    state = initial_state(ref)
    for x in spectra(300):
        _, state = full.run(None, {'spec': x, 'state_in': state})
    times = np.zeros(377, dtype=np.float64)
    full.lib.dpdf_profile_get.argtypes = [ct.POINTER(ct.c_double)]
    full.lib.dpdf_profile_get(times.ctypes.data_as(ct.POINTER(ct.c_double)))
    nodes = json.loads(Path('models/full_c/profile_nodes.json').read_text())
    result = sorted(zip(nodes, (times / 300 / 1e6).tolist()), key=lambda x: -x[1])
    print(json.dumps(result[:30], indent=2))
    Path('results/full_profile.json').write_text(json.dumps(result, indent=2)+'\n')
    full.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    run() if a.run else instrument()
