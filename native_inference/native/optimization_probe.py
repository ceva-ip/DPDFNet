"""Compare preserved and candidate native libraries on identical recurrent streams."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from statistics import median

from extended_probe import ExtendedModel, CONFIGS
from full_probe import timings
from probe import cpu_name, initial_state, session, spectra


def exact_recurrent_parity(left, right, frames):
    left_state = initial_state(left)
    right_state = initial_state(right)
    for index, frame in enumerate(frames):
        left_output, left_state = left.run(None, {'spec': frame, 'state_in': left_state})
        right_output, right_state = right.run(None, {'spec': frame, 'state_in': right_state})
        if left_output.tobytes() != right_output.tobytes():
            raise AssertionError(f'Output differs at frame {index}')
        if left_state.tobytes() != right_state.tobytes():
            raise AssertionError(f'State differs at frame {index}')
    return {'frames': len(frames), 'output_and_state_bit_identical': True}


def summarize(timing):
    result = {}
    for mode in ('baseline', 'candidate'):
        result[mode] = {
            'median_mean_ms': median(run['mean_ms'] for run in timing[mode]),
            'median_p50_ms': median(run['p50_ms'] for run in timing[mode]),
            'over_10ms_total': sum(run['over_10ms'] for run in timing[mode]),
        }
    for metric in ('median_mean_ms', 'median_p50_ms'):
        baseline = result['baseline'][metric]
        result[f'{metric}_saved_percent'] = (baseline-result['candidate'][metric])/baseline*100
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--baseline-build', type=Path, required=True)
    parser.add_argument('--candidate-build', type=Path, required=True)
    parser.add_argument('--config', default='fc_and_1x1_8', choices=CONFIGS)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--paced-repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = session(args.model)
    config = CONFIGS[args.config]
    models = {
        'baseline': ExtendedModel(reference, *config, build=args.baseline_build, weights=args.weights),
        'candidate': ExtendedModel(reference, *config, build=args.candidate_build, weights=args.weights),
    }
    parity = exact_recurrent_parity(models['baseline'], models['candidate'], spectra(500))
    continuous = timings(models, spectra(1100), args.repeats, False)
    paced = timings(models, spectra(1100), args.paced_repeats, True)
    report = {
        'config': args.config,
        'environment': {'cpu': cpu_name(), 'platform': platform.platform(), 'threads': 1},
        'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        'artifacts': {
            name: hashlib.sha256((build / 'libdpdf_full.so').read_bytes()).hexdigest()
            for name, build in [('baseline', args.baseline_build), ('candidate', args.candidate_build)]
        },
        'parity': parity,
        'summary': {'continuous': summarize(continuous), 'paced': summarize(paced)},
        'continuous': continuous,
        'paced': paced,
    }
    for model in models.values():
        model.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
