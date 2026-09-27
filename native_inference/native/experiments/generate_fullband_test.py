"""Run pinned official EARS-WHAM_v2 generator, skipping only train/valid.

The upstream script resets NumPy's RNG to 42 immediately before the test
split, so omitting preceding splits preserves its test-generation sequence.
The downloaded original is kept unchanged alongside the effective script.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

root = Path('/bench/scratch/fullband')
source = root/'generate_ears_wham.py'
original = source.read_text()
needle = 'for subset in ["train", "valid"]:'
assert original.count(needle) == 1
effective = original.replace(needle, 'for subset in []:  # Test-only invocation; upstream resets RNG before test')
script = root/'generate_ears_wham_test_only.py'
script.write_text(effective)
metadata = {'upstream_revision': json.loads((root/'ears_tree.json').read_text())['sha'],
            'original_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'effective_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
            'patch': 'Skip train/valid loop only; preserve upstream test code and defaults',
            'command': [sys.executable, str(script), '--data_dir', str(root/'data')]}
(root/'generation_provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
os.chdir(root)
subprocess.run(metadata['command'], check=True)
