"""Measure changed INT8 grids without claiming byte-exact recurrent parity.

Accepts the same arguments as latency_probe.py. All timing samples remain
unfiltered; the numerical comparison is not a speech-quality acceptance test.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import latency_probe
from probe import initial_state


def numerical_difference(baseline, candidate, frames):
    states = [initial_state(model) for model in (baseline, candidate)]
    maximum = [0., 0.]
    squared = [0., 0.]
    counts = [0, 0]
    for frame in frames:
        result = []
        for i, model in enumerate((baseline, candidate)):
            output, states[i] = model.run(None, {'spec': frame, 'state_in': states[i]})
            if not np.isfinite(output).all() or not np.isfinite(states[i]).all():
                raise AssertionError('Nonfinite output or recurrent state')
            result.append((output, states[i]))
        for i in range(2):
            diff = result[0][i].astype(np.float64)-result[1][i]
            maximum[i] = max(maximum[i], float(np.abs(diff).max()))
            squared[i] += float(np.sum(diff*diff))
            counts[i] += diff.size
    return {'exact': False, 'frames': len(frames), 'finite': True,
            'max_abs_output_state': maximum,
            'rms_output_state': [(v/n)**.5 for v, n in zip(squared, counts)],
            'note': 'Changed quantization; numerical differences are not a perceptual acceptance test.'}


if __name__ == '__main__':
    latency_probe.main(comparison=numerical_difference)
