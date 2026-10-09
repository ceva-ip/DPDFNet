"""48 kHz mono WAV example with causal FFT and delay-aligned output."""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from dpdfnet import Model


def window():
    n = np.arange(960, dtype=np.float64)
    return np.sin(0.5 * np.pi * np.sin(np.pi * (n + 0.5) / 960)**2).astype(np.float32)


def spectra(audio):
    count = (audio.size + 479) // 480 + 6
    padded = np.pad(audio, (480, count * 480 - audio.size))
    win = window()
    for index in range(count):
        value = np.fft.rfft(padded[index * 480:index * 480 + 960] * win)
        result = np.empty((481, 2), dtype=np.float32)
        result[:, 0], result[:, 1] = value.real, value.imag
        yield result


def enhance(model, audio):
    model.reset()
    win = window()
    overlap = np.zeros(960, dtype=np.float32)
    chunks = []
    for spectrum in spectra(audio):
        result = model.process(spectrum)
        time_frame = np.fft.irfft(result[:, 0] + 1j * result[:, 1], n=960).astype(np.float32) * win
        overlap[:480] = overlap[480:]; overlap[480:] = 0
        overlap += time_frame
        chunks.append(overlap[:480].copy())
    return np.concatenate(chunks)[2400:2400 + audio.size]


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=int, choices=(2, 8), default=8)
    parser.add_argument("--library", type=Path, default=root / "build/libdpdfnet.so")
    args = parser.parse_args()
    audio, rate = sf.read(args.input, dtype="float32", always_2d=True)
    if rate != 48000 or audio.shape[1] != 1 or not np.isfinite(audio).all():
        raise ValueError("Input must be finite 48 kHz mono audio; no implicit resampling")
    weights = root / "models" / ("dpdfnet%d_48khz_hr" % args.model) / "weights.f32"
    with Model(args.library, weights, args.model) as model:
        output = enhance(model, audio[:, 0])
    sf.write(args.output, output, rate, subtype="FLOAT")


if __name__ == "__main__":
    main()
