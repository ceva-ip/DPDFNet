"""Diagnose Python wrapper pauses and compare calls with preallocated buffers.

GC stays enabled in both modes. These diagnostic runs supplement the unfiltered
latency benchmark; they never replace or remove its samples.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from extended_probe import CONFIGS, ExtendedModel
from probe import initial_state, ptr, session, spectra, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'weights', 'baseline-build', 'candidate-build', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    reference = session(args.model)
    models = {name: ExtendedModel(reference, *CONFIGS['fc_and_1x1_8'], build=build, weights=args.weights)
              for name, build in [('baseline', args.baseline_build), ('candidate', args.candidate_build)]}
    frames = spectra(1100)
    inputs = [ptr(frame) for frame in frames]
    states = {name: initial_state(model) for name, model in models.items()}
    outputs = {name: np.empty((1, 1, 481, 2), dtype=np.float32) for name in models}
    state_ptrs = {name: ptr(state) for name, state in states.items()}
    output_ptrs = {name: ptr(output) for name, output in outputs.items()}

    def native(name, index):
        model = models[name]
        rc = model.lib.dpdf_model_process(model.handle, inputs[index], state_ptrs[name],
                                         output_ptrs[name], state_ptrs[name])
        if rc:
            raise RuntimeError(f'Native model returned {rc}')

    active = None
    gc_start = 0
    events = []

    def observe(phase, info):
        nonlocal gc_start
        if phase == 'start':
            gc_start = time.perf_counter_ns()
        else:
            events.append({'context': active, 'generation': info['generation'],
                           'duration_ms': (time.perf_counter_ns()-gc_start)/1e6,
                           'collected': info['collected']})

    report = {'method': 'Diagnostic continuous AB/BA comparisons; GC enabled in both allocating and preallocated modes.',
              'artifacts': {name: hashlib.sha256((build/'libdpdf_full.so').read_bytes()).hexdigest()
                            for name, build in [('baseline', args.baseline_build), ('candidate', args.candidate_build)]},
              'preallocated_parity_frames': 128, 'runs': []}
    try:
        for name, model in models.items():
            usual = initial_state(model)
            for index in range(128):
                output, usual = model.run(None, {'spec': frames[index], 'state_in': usual})
                native(name, index)
                if output.tobytes() != outputs[name].tobytes() or usual.tobytes() != states[name].tobytes():
                    raise AssertionError(f'Preallocated output/state differs: {name}, frame {index}')
        gc.callbacks.append(observe)
        for mode in ('allocating', 'preallocated'):
            for repeat in range(args.repeats):
                usual = {name: initial_state(model) for name, model in models.items()}
                for name, model in models.items():
                    states[name][...] = initial_state(model)
                timings = {name: [] for name in models}
                slow = []
                for index, frame in enumerate(frames):
                    names = list(models)
                    if (index+repeat) % 2:
                        names.reverse()
                    for name in names:
                        active = {'mode': mode, 'repeat': repeat, 'name': name, 'frame': index}
                        event_start = len(events)
                        begin = time.perf_counter_ns()
                        if mode == 'allocating':
                            _, usual[name] = models[name].run(None, {'spec': frame, 'state_in': usual[name]})
                        else:
                            native(name, index)
                        elapsed = (time.perf_counter_ns()-begin)/1e6
                        active = None
                        if index >= 100:
                            timings[name].append(elapsed)
                            if elapsed > 4:
                                slow.append({'name': name, 'frame': index, 'wall_ms': elapsed,
                                             'gc_ms': sum(e['duration_ms'] for e in events[event_start:])})
                run = {'mode': mode, 'repeat': repeat, 'implementations': {name: summarize(v) for name, v in timings.items()},
                       'calls_over_4ms': slow}
                report['runs'].append(run)
                print(json.dumps(run), flush=True)
        report['gc_events'] = events
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    finally:
        if observe in gc.callbacks:
            gc.callbacks.remove(observe)
        for model in models.values():
            model.close()


if __name__ == '__main__':
    main()
