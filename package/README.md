# dpdfnet

CPU-only ONNX inference package for DPDFNet speech enhancement.

## Installation

```bash
pip install dpdfnet
```

## Requirements

- Python `>=3.11`
- OS support for `soundfile` / `libsndfile`

Runtime dependencies are installed automatically:
- `numpy`
- `librosa`
- `soundfile`
- `onnxruntime`
- `filelock`
- `tqdm`

## Supported Audio Formats

The following input formats are supported out of the box (via `soundfile`/libsndfile):

| Format | Extensions |
|--------|-----------|
| WAV | `.wav` |
| FLAC | `.flac` |
| Ogg Vorbis | `.ogg` |
| AIFF | `.aiff`, `.aif` |
| AU/SND | `.au`, `.snd` |

MP3 and other compressed formats require the optional `pydub` dependency and
[ffmpeg](https://ffmpeg.org/) on your PATH:

```bash
pip install 'dpdfnet[mp3]'
# also install ffmpeg, e.g.:
#   Ubuntu/Debian:  sudo apt install ffmpeg
#   macOS:          brew install ffmpeg
#   Windows:        https://ffmpeg.org/download.html
```

Once installed, these additional formats are supported:

| Format | Extensions |
|--------|-----------|
| MP3 | `.mp3` |
| AAC / M4A | `.aac`, `.m4a` |
| WMA | `.wma` |
| Opus | `.opus` |

Output is always written as PCM16 `.wav` regardless of the input format.

## CLI

Show help:

```bash
dpdfnet --help
```

Commands:

1. `dpdfnet models`
- List supported models and local availability.

2. `dpdfnet enhance <input> <output.wav> [--model <name>] [--attn-limit-db DB] [-v|--verbose]`
- Enhance one audio file (any supported format; output is always `.wav`).

3. `dpdfnet enhance-dir <input_dir> <output_dir> [--model <name>] [--workers N] [--attn-limit-db DB] [-v|--verbose]`
- Enhance all supported audio files in a directory (non-recursive).
- Files are processed concurrently; `--workers` sets the thread count (default: CPU count).

4. `dpdfnet download [model] [--force|--refresh] [-q|--quiet | -v|--verbose]`
- Download all models when `model` is omitted, or one model when provided.

CLI examples:

```bash
# Enhance one file
dpdfnet enhance noisy.wav enhanced.wav --model dpdfnet4 --attn-limit-db 12

# Enhance a directory (uses all CPU cores by default)
dpdfnet enhance-dir ./noisy_wavs ./enhanced_wavs --model dpdfnet2 --attn-limit-db 12

# Enhance a directory with a fixed worker count
dpdfnet enhance-dir ./noisy_wavs ./enhanced_wavs --model dpdfnet2 --workers 4 --attn-limit-db 12

# Download models
dpdfnet download
dpdfnet download dpdfnet8
dpdfnet download dpdfnet4 --force
```

## Python API

Top-level exports:
- `dpdfnet.enhance`
- `dpdfnet.enhance_file`
- `dpdfnet.available_models`
- `dpdfnet.download`

In-memory enhancement:

```python
import soundfile as sf
import dpdfnet

audio, sr = sf.read("noisy.wav")
enhanced = dpdfnet.enhance(audio, sample_rate=sr, model="dpdfnet4", attn_limit_db=12)
sf.write("enhanced.wav", enhanced, sr)
```

Enhance one file:

```python
import dpdfnet

out_path = dpdfnet.enhance_file("noisy.wav", model="dpdfnet2", attn_limit_db=12)
print(out_path)
```

Model listing:

```python
import dpdfnet

for row in dpdfnet.available_models():
    print(row["name"], row["ready"], row["cached"])
```

Download models via API:

```python
import dpdfnet

dpdfnet.download()
dpdfnet.download("dpdfnet4")
```

### Real-time Microphone Enhancement

Install `sounddevice` (not included in `dpdfnet` dependencies):

```bash
pip install sounddevice
```

`StreamEnhancer` processes audio chunk-by-chunk, preserving RNN state across
calls.  Any chunk size works; enhanced samples are returned as soon as enough
data has accumulated for the first model frame (20 ms).

```python
import numpy as np
import sounddevice as sd
import dpdfnet

INPUT_SR   = 48000
# Use one model hop (10 ms) as the block size so process() returns
# exactly one hop's worth of enhanced audio on every callback.
BLOCK_SIZE = int(INPUT_SR * 0.010)   # 480 samples at 48 kHz

enhancer = dpdfnet.StreamEnhancer(model="dpdfnet2_48khz_hr")

def callback(indata, outdata, frames, time, status):
    mono_in = indata[:, 0] if indata.ndim > 1 else indata.ravel()
    enhanced = enhancer.process(mono_in, sample_rate=INPUT_SR)
    n = min(len(enhanced), frames)
    outdata[:n, 0] = enhanced[:n]
    if n < frames:
        outdata[n:] = 0.0   # silence while the first window accumulates

with sd.Stream(
    samplerate=INPUT_SR,
    blocksize=BLOCK_SIZE,
    channels=1,
    dtype="float32",
    callback=callback,
):
    print("Enhancing microphone input - press Ctrl+C to stop")
    try:
        while True:
            sd.sleep(100)
    except KeyboardInterrupt:
        pass

# Optional: drain the final partial window at the end of a recording
tail = enhancer.flush()
```

> **Notes:**
> 
> **Latency** - the first enhanced output arrives after one full model window
>   (~20 ms) has been buffered.  All subsequent blocks are returned with ~10 ms
>   additional delay.
> **Sample rate** - `StreamEnhancer` resamples internally.  Pass your device's
>   native rate as `sample_rate`; the return value is at the same rate.
> **Block size** - using `BLOCK_SIZE = int(SR * 0.010)` (one model hop) gives
>   one enhanced block per callback.  Other sizes also work but may produce empty
>   returns while the buffer fills.
> **Multiple streams** - create a separate `StreamEnhancer` per stream.  Call
>   `enhancer.reset()` between independent audio segments to clear RNN state.

## Links

- Homepage: https://github.com/ceva-ip/DPDFNet
- Issues: https://github.com/ceva-ip/DPDFNet/issues
