#!/usr/bin/env python3
"""Acquire small official 48 kHz EARS examples and make deterministic streams.

Audio stays in the chosen local directory. This is an independent regression
suite, not a reconstruction of validation/reference_streams.json. Dependencies:
NumPy and ffmpeg/ffprobe. Research examples: https://sp-uhh.github.io/ears_dataset/
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import numpy as np

BASE = "https://www2.informatik.uni-hamburg.de/sp/audio/publications/interspeech2024-ears/files/"
SPEECH = ["p002_emo_adoration_sentences", "p008_emo_contentment_sentences",
          "p010_emo_cuteness_sentences", "p011_emo_anger_sentences", "p012_rainbow_05_whisper",
          "p014_rainbow_04_loud", "p016_rainbow_03_regular", "p017_rainbow_08_fast",
          "p018_vegetative_eating", "p019_vegetative_yawning", "p020_freeform_speech_01"]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def spectra(audio):
    count = (len(audio) + 479) // 480 + 6
    padded = np.pad(audio, (480, count * 480 - len(audio)))
    n = np.arange(960, dtype=np.float64)
    win = np.sin(.5 * np.pi * np.sin(np.pi * (n + .5) / 960) ** 2).astype(np.float32)
    result = np.empty((count, 481, 2), dtype="<f4")
    for i in range(count):
        value = np.fft.rfft(padded[i * 480:i * 480 + 960] * win)
        result[i, :, 0], result[i, :, 1] = value.real, value.imag
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    raw = root / "downloads"; raw.mkdir(exist_ok=True)
    records, sources, clean = [], [], []
    urls = ["ears/" + name + ".wav" for name in SPEECH] + [
        "ears-wham/noisy/p102/00252_4.9dB.wav", "ears-wham/clean/p102/00252_4.9dB.wav",
        "ears-blind/noisy/00620.wav", "ears-reverb/noisy/p102/00328_0.64.wav"]
    for i, relative in enumerate(urls):
        path = raw / f"{i:02d}-{Path(relative).name}"
        if not path.exists():
            with urllib.request.urlopen(BASE + relative, timeout=90) as response:
                path.write_bytes(response.read())
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]))
        stream = probe["streams"][0]
        if int(stream["sample_rate"]) != 48000: raise ValueError("Expected native 48 kHz")
        data = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "48000", "-f", "f32le", "-"])
        audio = np.frombuffer(data, dtype="<f4").copy()
        if not np.isfinite(audio).all(): raise ValueError("Nonfinite source")
        sources.append({"url": BASE + relative, "download_sha256": sha(path),
                        "native_sample_rate": 48000, "decoded_samples": len(audio)})
        clean.append(audio)

    def save(name, scenario, audio, recipe):
        frames = spectra(np.asarray(audio, dtype=np.float32))
        path = root / (name + ".f32"); frames.tofile(path)
        records.append({"id": name, "scenario": scenario, "path": str(path),
                        "samples": len(audio), "hops": len(frames), "sha256": sha(path), "recipe": recipe})

    for i, audio in enumerate(clean): save(f"original-{i:02d}", "recorded", audio, {"source": i})
    for i, audio in enumerate(clean[:11]):
        save(f"low-{i:02d}", "low_level", audio * np.float32(10 ** (-50 / 20)), {"source": i, "gain_db": -50})
    rng = np.random.default_rng(20261026)
    for i, audio in enumerate(clean[:9]):
        for snr in (-5, 5, 15):
            noise = rng.standard_normal(len(audio)).astype(np.float32)
            scale = np.sqrt(np.mean(audio.astype(np.float64) ** 2) / np.mean(noise.astype(np.float64) ** 2)) * 10 ** (-snr / 20)
            save(f"mixture-{i:02d}-{snr:+03d}", "speech_noise", audio + noise * np.float32(scale),
                 {"source": i, "snr_db": snr, "seed_stream": 20261026})
    count = 120 * 48000
    white = rng.standard_normal(count)
    frequencies = np.fft.rfftfreq(count)
    pink = np.fft.irfft(np.fft.rfft(white) / np.sqrt(np.maximum(frequencies, 1 / count)), n=count)
    t = np.arange(count) / 48000
    mechanical = np.sin(2*np.pi*50*t) + .3*np.sin(2*np.pi*137*t) + .1*white
    for name, audio in (("white", white), ("pink", pink), ("mechanical", mechanical)):
        audio = audio / np.sqrt(np.mean(audio**2)) * .05
        save("long-" + name, "long_noise", audio, {"seconds": 120, "rms": .05, "seed_stream": 20261026})
    save("silence", "silence", np.zeros(20 * 48000, dtype=np.float32), {})
    for level in (.9, 1e-6, 1e-20):
        t = np.arange(10 * 48000) / 48000
        save(f"tone-{level:g}", "tone", level*np.sin(2*np.pi*997*t), {"amplitude": level, "frequency": 997})
    impulses = np.zeros(10*48000, dtype=np.float32); impulses[::4801] = .9
    save("impulses", "impulse", impulses, {"spacing_samples": 4801})
    speech_silence = np.concatenate([np.zeros(48000, dtype=np.float32), clean[0], np.zeros(10*48000, dtype=np.float32)])
    save("speech-to-silence", "transition", speech_silence, {"source": 0, "post_silence_seconds": 10})
    for gain in (.01, .5, 8):
        save(f"gain-{gain:g}", "level_change", clean[6]*np.float32(gain), {"source": 6, "gain": gain})
    assert len(records) == 65, len(records)
    manifest = {"schema_version": 1, "source_page": "https://sp-uhh.github.io/ears_dataset/",
                "attribution": "Richter et al., EARS, Interspeech 2024; official research examples, CC BY-NC 4.0",
                "historical_suite": False, "numpy_version": np.__version__, "sample_rate": 48000,
                "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
                "sources": sources, "cases": records, "hops_per_model": sum(r["hops"] for r in records)}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"cases": len(records), "hops_per_model": manifest["hops_per_model"]}))


if __name__ == "__main__": main()
