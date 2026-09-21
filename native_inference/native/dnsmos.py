"""Pinned local DNSMOS P.835/P.808 scoring compatible with Microsoft's script."""
import hashlib
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort


SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01
DNSMOS_KEYS = ('p835_sig', 'p835_bak', 'p835_ovrl', 'p808_mos')


class DNSMOS:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        primary = model_dir / 'sig_bak_ovr.onnx'
        p808 = model_dir / 'model_v8.onnx'
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.primary = ort.InferenceSession(str(primary), options,
                                            providers=['CPUExecutionProvider'])
        self.p808 = ort.InferenceSession(str(p808), options,
                                         providers=['CPUExecutionProvider'])
        self.metadata = {
            'implementation': 'Microsoft DNS-Challenge dnsmos_local.py compatible; regular/non-personalized MOS',
            'source': 'https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS',
            'sample_rate': SAMPLING_RATE,
            'window_seconds': INPUT_LENGTH,
            'hop_seconds': 1,
            'librosa_version': librosa.__version__,
            'models': {
                primary.name: hashlib.sha256(primary.read_bytes()).hexdigest(),
                p808.name: hashlib.sha256(p808.read_bytes()).hexdigest(),
            },
        }

    @staticmethod
    def _mel(audio):
        mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE,
                                             n_fft=321, hop_length=160,
                                             n_mels=120)
        return ((librosa.power_to_db(mel, ref=np.max)+40)/40).T

    @staticmethod
    def _calibrate(sig, bak, ovrl):
        # Regular (non-personalized) polynomial calibration from dnsmos_local.py.
        return (
            np.poly1d([-0.08397278, 1.22083953, 0.0052439])(sig),
            np.poly1d([-0.13166888, 1.60915514, -0.39604546])(bak),
            np.poly1d([-0.06766283, 1.11546468, 0.04602535])(ovrl),
        )

    def score(self, audio, sample_rate):
        audio = np.asarray(audio, dtype=np.float32)
        assert audio.ndim == 1 and audio.size and np.isfinite(audio).all()
        if sample_rate != SAMPLING_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate,
                                     target_sr=SAMPLING_RATE).astype(np.float32)
        window = int(INPUT_LENGTH*SAMPLING_RATE)
        while audio.size < window:
            audio = np.append(audio, audio)
        hops = int(np.floor(audio.size/SAMPLING_RATE)-INPUT_LENGTH)+1
        assert hops > 0
        values = []
        for index in range(hops):
            start = index*SAMPLING_RATE
            segment = audio[start:start+window]
            assert segment.size == window
            primary_input = {'input_1': segment[np.newaxis, :].astype(np.float32)}
            sig_raw, bak_raw, ovrl_raw = self.primary.run(None, primary_input)[0][0]
            sig, bak, ovrl = self._calibrate(sig_raw, bak_raw, ovrl_raw)
            mel = self._mel(segment[:-160]).astype(np.float32)[np.newaxis, :, :]
            p808 = self.p808.run(None, {'input_1': mel})[0][0][0]
            values.append((sig, bak, ovrl, p808))
        means = np.mean(np.asarray(values, dtype=np.float64), axis=0)
        result = {key: float(value) for key, value in zip(DNSMOS_KEYS, means)}
        result['num_hops'] = hops
        return result
