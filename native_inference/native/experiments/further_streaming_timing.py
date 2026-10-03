"""Matched, preallocated single-thread W7A8 calls and CPU accounting.

Benchmarks must run without concurrent build/validation work. Paired cadence
has two implementations per hop; standalone cadence has one. No samples are
filtered. Python bookkeeping, FFT and audio-device work are outside calls.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
from statistics import median
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'native'))
from extended_probe import CONFIGS, ExtendedModel
from probe import cpu_name, initial_state, ptr, session, spectra, summarize
from optimization_probe import exact_recurrent_parity


def run(models, frames, repeats, paced, warmup):
    inputs = [ptr(frame) for frame in frames]
    states = {name: initial_state(model) for name, model in models.items()}
    outputs = {name: np.empty((1, 1, 481, 2), np.float32) for name in models}
    state_ptrs = {name: ptr(state) for name, state in states.items()}
    output_ptrs = {name: ptr(output) for name, output in outputs.items()}
    calls = {name: model.lib.dpdf_model_process for name, model in models.items()}
    results = []
    for repeat in range(repeats):
        for name, model in models.items():
            states[name][...] = initial_state(model)
        timings = {name: {'wall': [], 'thread_cpu': [], 'process_cpu': [],
                          'completion': [], 'switches': 0, 'late_calls': []} for name in models}
        start = time.perf_counter()
        for index in range(len(frames)):
            deadline = start + index * .01
            if paced:
                delay = deadline - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            names = list(models)
            if (index + repeat) % 2:
                names.reverse()
            for name in names:
                before = resource.getrusage(resource.RUSAGE_THREAD)
                cpu_start = time.process_time_ns()
                thread_start = time.thread_time_ns()
                wall_start = time.perf_counter_ns()
                rc = calls[name](models[name].handle, inputs[index], state_ptrs[name],
                                 output_ptrs[name], state_ptrs[name])
                wall = (time.perf_counter_ns() - wall_start) / 1e6
                thread_cpu = (time.thread_time_ns() - thread_start) / 1e6
                process_cpu = (time.process_time_ns() - cpu_start) / 1e6
                completion = (time.perf_counter() - deadline) * 1000
                after = resource.getrusage(resource.RUSAGE_THREAD)
                if rc:
                    raise RuntimeError(f'{name} process returned {rc}')
                if index < warmup:
                    continue
                record = timings[name]
                record['wall'].append(wall)
                record['thread_cpu'].append(thread_cpu)
                record['process_cpu'].append(process_cpu)
                if paced:
                    record['completion'].append(max(0, completion))
                switches = (after.ru_nvcsw + after.ru_nivcsw - before.ru_nvcsw - before.ru_nivcsw)
                record['switches'] += int(switches > 0)
                if wall > 10 or (paced and completion > 10):
                    record['late_calls'].append({'frame': index, 'wall_ms': wall,
                        'process_cpu_ms': process_cpu, 'completion_after_release_ms': completion,
                        'caller_context_switches': switches})
        report = {'repeat': repeat, 'paced': paced, 'implementations': {}}
        for name, values in timings.items():
            item = {metric: summarize(values[metric]) for metric in ('wall', 'thread_cpu', 'process_cpu')}
            if paced:
                item['completion_after_release'] = summarize(values['completion'])
            item['calls_with_caller_context_switches'] = values['switches']
            item['late_calls'] = values['late_calls']
            report['implementations'][name] = item
        results.append(report)
        print(json.dumps({'repeat': repeat, 'paced': paced,
              'means_ms': {name: {'wall': item['wall']['mean_ms'],
                                 'process_cpu': item['process_cpu']['mean_ms']}
                           for name, item in report['implementations'].items()}}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-build', 'candidate-build', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/dpdfnet8_48khz_hr.onnx')
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/rework8/weights.f32')
    parser.add_argument('--frames', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--paced-repeats', type=int, default=4)
    parser.add_argument('--standalone-paced-repeats', type=int, default=4)
    args = parser.parse_args()
    if min(args.frames, args.repeats, args.paced_repeats, args.standalone_paced_repeats) <= 0 or args.warmup < 0:
        parser.error('Positive frames/repeats and nonnegative warmup required')
    reference = session(args.model)
    builds = {'baseline': args.baseline_build, 'candidate': args.candidate_build}
    models = {name: ExtendedModel(reference, *CONFIGS['fc_and_1x1_8'], build=build, weights=args.weights)
              for name, build in builds.items()}
    try:
        frames = spectra(args.frames + args.warmup)
        report = {'environment': {'cpu': cpu_name(), 'platform': platform.platform(),
                      'affinity': sorted(os.sched_getaffinity(0))},
                  'method': 'Preallocated single-thread C calls; balanced AB/BA; unfiltered samples; caller and process CPU.',
                  'parity': exact_recurrent_parity(models['baseline'], models['candidate'], spectra(500)),
                  'config': 'fc_and_1x1_8; fitted W7A8 baseline',
                  'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                  'artifacts': {name: hashlib.sha256((build / 'libdpdf_full.so').read_bytes()).hexdigest()
                                for name, build in builds.items()},
                  'source_manifests': {name: json.loads((ROOT / ('scratch/further_optimization2' if build.name.startswith('further2_') else 'scratch/further_optimization') /
                      build.name.removeprefix('further2_').removeprefix('further_') /
                      'source_manifest.json').read_text()) for name, build in builds.items()},
                  'owned_bytes': {name: model.owned_bytes for name, model in models.items()},
                  'timed_frames_per_run': args.frames, 'warmup': args.warmup,
                  'continuous': run(models, frames, args.repeats, False, args.warmup),
                  'paced': run(models, frames, args.paced_repeats, True, args.warmup),
                  'standalone_paced': []}
        for repeat in range(args.standalone_paced_repeats):
            names = list(models)
            if repeat % 2:
                names.reverse()
            for name in names:
                item = run({name: models[name]}, frames, 1, True, args.warmup)[0]
                item['repeat'] = repeat
                report['standalone_paced'].append(item)
        report['summary'] = {}
        for mode in ('continuous', 'paced', 'standalone_paced'):
            rows = report[mode]
            values = {name: {'median_run_mean_ms': median(item['implementations'][name]['wall']['mean_ms']
                       for item in rows if name in item['implementations']),
                       'median_process_cpu_ms': median(item['implementations'][name]['process_cpu']['mean_ms']
                       for item in rows if name in item['implementations'])} for name in models}
            values['mean_reduction_percent'] = 100 * (1 - values['candidate']['median_run_mean_ms'] /
                                                     values['baseline']['median_run_mean_ms'])
            report['summary'][mode] = values
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report['summary']), flush=True)
    finally:
        for model in models.values():
            model.close()


if __name__ == '__main__':
    main()
