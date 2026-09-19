"""Reproducible graph-only DPDFNet streaming probe; not an audio quality test.

No PyTorch dependency. Feed each output state into the next frame, seed
normalizers from model metadata, and time separately from ORT profiling.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import onnx
import onnxruntime as ort


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_name():
    if Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    return platform.processor()


def session(path, profile_prefix=None):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile_prefix:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def initial_state(sess):
    meta = sess.get_modelmeta().custom_metadata_map
    assert int(meta['sample_rate']) == 48000
    assert int(meta['hop_length']) == 480
    assert int(meta['n_fft']) == 960
    state = np.zeros(int(meta['state_size']), dtype=np.float32)
    offset = 0
    for prefix in ('erb', 'spec'):
        values = np.asarray([float(v) for v in meta[f'{prefix}_norm_init'].split(',')], dtype=np.float32)
        size = int(meta[f'{prefix}_norm_state_size'])
        assert size == values.size and offset + size <= state.size
        state[offset:offset + size] = values
        offset += size
    assert sess.get_inputs()[0].shape == [1, 1, 481, 2]
    assert sess.get_inputs()[1].shape == [state.size]
    return state


def spectra(count):
    """Deterministic continuous synthetic PCM on HushMic's causal FFT grid.

    Sine mixture, amplitude changes, noise, impulses and a silence interval.
    This exercises recurrence but is NOT representative speech evaluation.
    FFT is prepared outside timing; no STFT/iSTFT cost in reported latency.
    """
    rng = np.random.default_rng(20260918)
    t = np.arange(count * 480, dtype=np.float64) / 48000
    pcm = ((0.08 + 0.06 * np.sin(2 * np.pi * 0.7 * t)) *
           (np.sin(2 * np.pi * 173 * t) + 0.3 * np.sin(2 * np.pi * 911 * t)) +
           0.02 * rng.standard_normal(t.size)).astype(np.float32)
    pcm[(t > 2) & (t < 2.5)] = 0
    pcm[::24000] += 0.5
    n = np.arange(960, dtype=np.float64)
    window = np.sin(0.5 * np.pi * np.sin(np.pi * (n + 0.5) / 960)**2).astype(np.float32)
    padded = np.pad(pcm, (480, 0))
    result = np.empty((count, 1, 1, 481, 2), dtype=np.float32)
    for i in range(count):
        spectrum = np.fft.rfft(padded[i * 480:i * 480 + 960] * window)
        result[i, 0, 0, :, 0] = spectrum.real
        result[i, 0, 0, :, 1] = spectrum.imag
    return result


class Runner:
    def __init__(self, sess, api):
        self.sess, self.api = sess, api
        self.states = [initial_state(sess), np.empty_like(initial_state(sess))]
        self.spec = np.empty((1, 1, 481, 2), dtype=np.float32)
        self.output = np.empty_like(self.spec)
        self.index = 0
        self.bindings = []
        if api == 'binding':
            for i in range(2):
                binding = sess.io_binding()
                binding.bind_cpu_input('spec', self.spec)
                binding.bind_cpu_input('state_in', self.states[i])
                for name, value in [('spec_e', self.output), ('state_out', self.states[1-i])]:
                    binding.bind_output(name, 'cpu', 0, np.float32, value.shape, value.ctypes.data)
                self.bindings.append(binding)

    def reset(self):
        self.states[0][:] = initial_state(self.sess)
        self.index = 0

    def run(self, spec):
        if self.api == 'binding':
            np.copyto(self.spec, spec)
            self.sess.run_with_iobinding(self.bindings[self.index])
            self.index = 1 - self.index
            return self.output, self.states[self.index]
        output, state = self.sess.run(['spec_e', 'state_out'], {
            'spec': spec, 'state_in': self.states[0]})
        self.states[0] = state
        return output, state


def timed_run(runner, frames, warmup, repeats, paced):
    runs = []
    for _ in range(repeats):
        runner.reset()
        elapsed = []
        late_starts = []
        start = time.perf_counter()
        for i, frame in enumerate(frames):
            target = start + i * 0.01
            if paced:
                remaining = target - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
            before = time.perf_counter_ns()
            out, state = runner.run(frame)
            duration = (time.perf_counter_ns() - before) / 1e6
            if i >= warmup:
                elapsed.append(duration)
                if paced:
                    late_starts.append(max(0.0, before / 1e9 - target) * 1000)
        if not np.isfinite(out).all() or not np.isfinite(state).all():
            raise ValueError('Non-finite output/state')
        times = np.asarray(elapsed)
        stats = {'mean_ms': float(times.mean()), 'rtf_mean': float(times.mean() / 10),
                 'p50_ms': float(np.percentile(times, 50)), 'p95_ms': float(np.percentile(times, 95)),
                 'p99_ms': float(np.percentile(times, 99)), 'max_ms': float(times.max()),
                 'over_10ms': int((times > 10).sum())}
        if paced:
            stats['start_lateness_p99_ms'] = float(np.percentile(late_starts, 99))
        runs.append(stats)
    return runs


def inventory(path):
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    inits = {x.name: x for x in model.graph.initializer}
    grus = []
    for node in model.graph.node:
        if node.op_type == 'GRU':
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            grus.append({'name': node.name, 'weights': list(inits[node.input[1]].dims),
                         'attributes': {k: v.decode() if isinstance(v, bytes) else v for k, v in attrs.items()}})
    return {'nodes': len(model.graph.node), 'operators': dict(Counter(n.op_type for n in model.graph.node)),
            'initializer_elements': sum(int(np.prod(x.dims)) for x in inits.values()),
            'gru_nodes': grus, 'opsets': {x.domain: x.version for x in model.opset_import}}


def profile(path, frames, prefix):
    sess = session(path, prefix)
    runner = Runner(sess, 'run')
    for frame in frames:
        runner.run(frame)
    trace_path = Path(sess.end_profiling())
    trace = json.loads(trace_path.read_text())
    totals, counts, names = defaultdict(float), Counter(), defaultdict(float)
    for event in trace:
        if event.get('cat') == 'Node' and event['name'].endswith('_kernel_time'):
            op = event['args'].get('op_name', 'unknown')
            totals[op] += event['dur']
            counts[op] += 1
            names[event['name']] += event['dur']
    total = sum(totals.values())
    return {'note': 'Instrumented kernel durations include profiling overhead and startup; not latency measurements.',
            'operators': [{'op': op, 'share_percent': 100 * us / total, 'calls': counts[op]}
                          for op, us in sorted(totals.items(), key=lambda x: -x[1])],
            'top_nodes': [{'name': name, 'share_percent': 100 * us / total}
                          for name, us in sorted(names.items(), key=lambda x: -x[1])[:20]]}


def compare(reference, candidate, frames, candidate_api):
    left, right = Runner(session(reference), 'run'), Runner(session(candidate), candidate_api)
    stats = {k: {'max_abs': 0.0, 'error_squared': 0.0, 'reference_squared': 0.0} for k in ('spectrum', 'state')}
    for frame in frames:
        for key, a, b in zip(stats, left.run(frame), right.run(frame)):
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Non-finite parity result')
            diff = a.astype(np.float64) - b
            stats[key]['max_abs'] = max(stats[key]['max_abs'], float(np.abs(diff).max()))
            stats[key]['error_squared'] += float(np.sum(diff**2))
            stats[key]['reference_squared'] += float(np.sum(a.astype(np.float64)**2))
    for item in stats.values():
        item['relative_rms'] = float(np.sqrt(item['error_squared'] / max(item['reference_squared'], 1e-30)))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--api', choices=['run', 'binding'], default='run')
    parser.add_argument('--frames', type=int, default=1000, help='Timed frames per repeat')
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--paced', action='store_true')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--reference', type=Path, help='Compare independent recurrent trajectories; not a quality score')
    args = parser.parse_args()
    if min(args.frames, args.repeats) <= 0 or args.warmup < 0:
        parser.error('frames/repeats must be positive; warmup must be nonnegative')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = spectra(args.frames + args.warmup)
    sess = session(args.model)
    result = {'model': args.model.name, 'sha256': digest(args.model), 'bytes': args.model.stat().st_size,
              'environment': {'platform': platform.platform(), 'cpu': cpu_name(), 'machine': platform.machine(),
                              'logical_cpus': os.cpu_count(), 'python': platform.python_version(),
                              'numpy': np.__version__, 'onnx': onnx.__version__, 'onnxruntime': ort.__version__},
              'config': {'api': args.api, 'frames': args.frames, 'warmup': args.warmup, 'repeats': args.repeats,
                         'paced': args.paced, 'threads': 1, 'hop_ms': 10, 'input': 'synthetic causal STFT; seed 20260918'},
              'state_elements': initial_state(sess).size, 'inventory': inventory(args.model)}
    result['runs'] = timed_run(Runner(sess, args.api), frames, args.warmup, args.repeats, args.paced)
    if args.profile:
        scratch = args.output.parent / 'scratch'
        scratch.mkdir(exist_ok=True)
        result['profile'] = profile(args.model, frames[:100], scratch / args.output.stem)
    if args.reference:
        result['reference_sha256'] = digest(args.reference)
        result['parity'] = compare(args.reference, args.model, frames, args.api)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('environment', 'runs', 'parity') if k in result}, indent=2))


if __name__ == '__main__':
    main()
