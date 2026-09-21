"""Byte-exact recurrent regression and independent-context concurrency checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from extended_probe import CONFIGS, ExtendedModel
from optimization_probe import exact_recurrent_parity
from probe import initial_state, session, spectra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--baseline-build', type=Path, required=True)
    parser.add_argument('--candidate-build', type=Path, required=True)
    parser.add_argument('--configs', nargs='+', choices=CONFIGS,
                        default=['compact_fp32', 'fc_and_1x1_16', 'fc_and_1x1_8'])
    parser.add_argument('--frames', type=int, default=500)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error('--frames must be positive')
    reference = session(args.model)
    frames = spectra(args.frames)
    report = {'model': str(args.model), 'parity': {}, 'concurrency': {},
              'artifacts': {name: hashlib.sha256((build/'libdpdf_full.so').read_bytes()).hexdigest()
                            for name, build in [('baseline', args.baseline_build),
                                                ('candidate', args.candidate_build)]}}
    for config in args.configs:
        models = [ExtendedModel(reference, *CONFIGS[config], build=build, weights=args.weights)
                  for build in (args.baseline_build, args.candidate_build)]
        try:
            report['parity'][config] = exact_recurrent_parity(*models, frames)
        finally:
            for model in models:
                model.close()

        # Each worker owns a distinct model context and recurrent state. Record
        # every output/state, not just the last frame; concurrent inputs differ.
        def stream(index):
            model = ExtendedModel(reference, *CONFIGS[config], build=args.candidate_build,
                                  weights=args.weights)
            state = initial_state(model)
            digest = hashlib.sha256()
            try:
                for frame in frames[:64]:
                    output, state = model.run(None, {'spec': frame*(index+1)/4,
                                                    'state_in': state})
                    digest.update(output.tobytes())
                    digest.update(state.tobytes())
                return digest.hexdigest()
            finally:
                model.close()

        expected = [stream(index) for index in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(stream, range(4)))
        if expected != actual:
            raise AssertionError(f'Independent concurrent contexts differ: {config}')
        report['concurrency'][config] = {'streams': 4, 'frames_each': min(64, len(frames)),
                                          'serial_and_concurrent_bit_identical': True,
                                          'output_and_state_sha256': actual}
        print(f'{config}: {len(frames)} recurrent frames and four concurrent contexts match', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
