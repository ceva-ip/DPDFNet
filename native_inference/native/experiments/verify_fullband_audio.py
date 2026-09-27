"""Verify persisted 48 kHz waveforms against the scored PCM hashes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf

root = Path('/bench')
data = json.loads((root/'results/fullband_ears_wham_v2_clips.json').read_text())
count = 0
for clip in data['clips']:
    for system in ('onnx', 'fp16', 'int8', 'w7a8'):
        path = root/'scratch/fullband/evaluation/audio'/system/clip['clip']['speaker']/(clip['clip']['id']+'.wav')
        audio, rate = sf.read(path, dtype='float32')
        assert rate == 48000 and audio.shape == (clip['samples'],) and np.isfinite(audio).all()
        assert hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest() == clip['systems'][system]['pcm_sha256']
        count += 1
assert count == 200
print(f'Verified all {count} saved enhanced waveforms: rate, length, finite PCM, and SHA256')
