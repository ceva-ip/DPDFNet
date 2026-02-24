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

## CLI

Show help:

```bash
dpdfnet --help
```

Commands:

1. `dpdfnet models`
- List supported models and local availability.

2. `dpdfnet enhance <input.wav> <output.wav> [--model <name>] [-v|--verbose]`
- Enhance one WAV file.

3. `dpdfnet enhance-dir <input_dir> <output_dir> [--model <name>] [-v|--verbose]`
- Enhance all `.wav` files in a directory (non-recursive).

4. `dpdfnet download [model] [--force|--refresh] [-q|--quiet | -v|--verbose]`
- Download all models when `model` is omitted, or one model when provided.

CLI examples:

```bash
# Enhance one file
dpdfnet enhance noisy.wav enhanced.wav --model dpdfnet4

# Enhance a directory
dpdfnet enhance-dir ./noisy_wavs ./enhanced_wavs --model dpdfnet2

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
enhanced = dpdfnet.enhance(audio, sample_rate=sr, model="dpdfnet4")
sf.write("enhanced.wav", enhanced, sr)
```

Enhance one file:

```python
import dpdfnet

out_path = dpdfnet.enhance_file("noisy.wav", model="dpdfnet2")
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

## Links

- Homepage: https://github.com/ceva-ip/DPDFNet
- Issues: https://github.com/ceva-ip/DPDFNet/issues
- Contributing guide: `CONTRIBUTING.md`
