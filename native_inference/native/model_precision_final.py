"""Final ONNX/FP32/selective-FP16/selective-INT8 comparison for one model."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from extended_probe import ExtendedModel
from full_probe import timings
from probe import cpu_name, session, spectra, whole_parity


FINAL_CONFIGS = {
    'native_fp32': (0, 0, 0),
    'selective_fp16': (3, 16, 7),
    'selective_int8': (4, 8, 7),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--memory-probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--frames', type=int, default=1000)
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmup < 0 or args.frames <= 0:
        parser.error('repeats/frames must be positive and warmup must be non-negative')

    reference = session(args.model)
    sessions = {'original_fp32': reference}
    for name, config in FINAL_CONFIGS.items():
        sessions[name] = ExtendedModel(reference, *config, build=args.build, weights=args.weights)

    validation_frames = spectra(300)
    parity = whole_parity(reference, sessions['native_fp32'], validation_frames, 'synthetic')
    benchmark_frames = spectra(args.warmup + args.frames)
    report = {
        'model': args.model_name,
        'environment': {'cpu': cpu_name(), 'platform': platform.platform()},
        'precision_scope': 'DPRNN + dense/grouped FC + 1x1 CNN; other CNN remains FP32',
        'config': {'timed_frames': args.frames, 'warmup': args.warmup,
                   'repeats': args.repeats, 'threads': 1},
        'configs': FINAL_CONFIGS,
        'artifacts': {
            str(args.model): sha256(args.model),
            str(args.build / 'libdpdf_full.so'): sha256(args.build / 'libdpdf_full.so'),
            str(args.weights): sha256(args.weights),
        },
        'fp32_parity': parity,
        'continuous': timings(sessions, benchmark_frames, args.repeats, False, args.warmup),
        'paced': timings(sessions, benchmark_frames, args.repeats, True, args.warmup),
        'memory': {},
        'memory_method': 'Fresh exec; /proc/self/status VmRSS and VmHWM',
    }
    for name, (tier, precision, mask) in FINAL_CONFIGS.items():
        command = [str(args.memory_probe), str(args.build / 'libdpdf_full.so'),
                   str(args.weights), str(tier), str(precision), str(mask)]
        report['memory'][name] = [
            json.loads(subprocess.check_output(command, text=True))
            for _ in range(args.repeats)
        ]
    for model in sessions.values():
        close = getattr(model, 'close', None)
        if close:
            close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
