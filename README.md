

<h1 align="center">DPDFNet: Boosting DeepFilterNet2 via Dual-Path RNN</h1>
<br></br>

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-orange)](https://ceva-ip.github.io/DPDFNet/)
[![arXiv Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b)](https://arxiv.org/abs/2512.16420)
[![Hugging Face Models](https://img.shields.io/badge/Hugging%20Face-Models-yellow)](https://huggingface.co/Ceva-IP/DPDFNet)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-yellowgreen)](https://huggingface.co/datasets/Ceva-IP/DPDFNet_EvalSet)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Ceva-IP/DPDFNetDemo)


</div>

<p align="center">
 <sub><em><strong>--- Official project for the DPDFNet paper ---</strong></em></sub>
</p>

<p align="center">
  <img src="figures/dpdfnet2_48khz_hr_gif.gif" width="688" alt="Noisy→Enhanced spectrogram slideshow" />
</p>

## Model Profile

### 16 kHz models

| Model | Params [M] | MACs [G] | TFLite Size [MB] | ONNX Size [MB] | Intended Use |
| --- | :---: | :---: | :---: | :---: | --- |
| baseline | 2.31 | 0.36 | 8.5 | 8.3 | Fastest / lowest resource usage |
| dpdfnet2 | 2.49 | 1.35 | 10.7 | 9.7 | Real-time / embedded devices |
| dpdfnet4 | 2.84 | 2.36 | 12.9 | 11.1 | Balanced performance |
| dpdfnet8 | 3.54 | 4.37 | 17.2 | 13.9 | Best enhancement quality |

### 48 kHz models

| Model | Params [M] | MACs [G] | TFLite Size [MB] | ONNX Size [MB] | Intended Use |
| --- | :---: | :---: | :---: | :---: | --- |
| dpdfnet2_48khz_hr | 2.58 | 2.42 | 11.6 | 10.0 | Balanced performance |
| dpdfnet8_48khz_hr | 3.63 | 7.17 | 18.7 | 14.2 | Best enhancement quality |

## Install the PyPI Package

For CPU-only ONNX inference using the packaged CLI and Python API:

```bash
pip install dpdfnet
```

### CLI Example

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

### Python API Example

```python
import soundfile as sf
import dpdfnet

# In-memory enhancement:
audio, sr = sf.read("noisy.wav")
enhanced = dpdfnet.enhance(audio, sample_rate=sr, model="dpdfnet4", attn_limit_db=12)
sf.write("enhanced.wav", enhanced, sr)

# Enhance one file:
out_path = dpdfnet.enhance_file("noisy.wav", model="dpdfnet2", attn_limit_db=12)
print(out_path)

# Model listing:
for row in dpdfnet.available_models():
    print(row["name"], row["ready"], row["cached"])

# Download models:
dpdfnet.download()				# All models
dpdfnet.download("dpdfnet4")	# Specific model
```

### Streaming (Real-Time) API

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
>   additional delay.\
> **Sample rate** - `StreamEnhancer` resamples internally.  Pass your device's
>   native rate as `sample_rate`; the return value is at the same rate.\
> **Block size** - using `BLOCK_SIZE = int(SR * 0.010)` (one model hop) gives
>   one enhanced block per callback.  Other sizes also work but may produce empty
>   returns while the buffer fills.\
> **Multiple streams** - create a separate `StreamEnhancer` per stream.  Call
>   `enhancer.reset()` between independent audio segments to clear RNN state.

## Run From Source

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Download models

Model files are not bundled in this repository.\
Download PyTorch checkpoints, TFLite, and ONNX models from Hugging Face:

```bash
pip install -U "huggingface_hub[cli]"

# create target dirs
mkdir -p model_zoo/{checkpoints,onnx,tflite}

# PyTorch checkpoints (HF path: checkpoints/* -> local: model_zoo/checkpoints/*)
hf download Ceva-IP/DPDFNet \
  --include "checkpoints/*.pth" \
  --local-dir model_zoo \

# ONNX models (HF path: onnx/* -> local: model_zoo/onnx/*)
hf download Ceva-IP/DPDFNet \
  --include "onnx/*.onnx" \
  --local-dir model_zoo \

# TFLite models (HF path: *.tflite at repo root -> local: model_zoo/tflite/*)
hf download Ceva-IP/DPDFNet \
  --include "*.tflite" \
  --local-dir model_zoo/tflite \
```

### 3) Run offline enhancement

Put one or more `*.wav` files in `./noisy_wavs`, then choose one:

#### Option A: `TFLite`

```bash
python -m tflite_model.infer_dpdfnet_tflite \
	--noisy_dir ./noisy_wavs \
	--enhanced_dir ./enhanced_wavs \
	--model_name dpdfnet4 \
	--workers 5 \
	--attn-limit-db 12
```

#### Option B: `ONNX`

```bash
python -m onnx_model.infer_dpdfnet_onnx \
  	--noisy_dir ./noisy_wavs \
	--enhanced_dir ./enhanced_wavs \
	--model_name dpdfnet4 \
	--workers 5 \
	--attn-limit-db 12
```

Enhanced files are written as:

```text
<original_stem>_<model_name>.wav
```

## Audio Samples & Demo

- Project page with examples: https://ceva-ip.github.io/DPDFNet/
- Gradio application: https://huggingface.co/spaces/Ceva-IP/DPDFNetDemo
- Hugging Face model hub: https://huggingface.co/Ceva-IP/DPDFNet
- Evaluation dataset used in the paper: https://huggingface.co/datasets/Ceva-IP/DPDFNet_EvalSet

## Real-Time Demo

![Real-time DPDFNet demo screenshot](figures/live_demo.png)

Run:

```bash
python -m real_time_demo
```

How it works:
- Captures microphone audio in streaming hops.
- Enhances each hop frame-by-frame with ONNX.
- Displays live noisy vs enhanced spectrograms.
- Allows you to control the noise‑reduction level during playback: `0` for the raw stream and `1` for the fully enhanced stream.
- Enables the use of AGC during playback.

To change model, edit `MODEL_NAME` near the top of `real_time_demo.py`.

## Troubleshooting / FAQ

#### `Q: Model files are missing (TFLite / ONNX / checkpoints)`
- Run the Hugging Face download commands from the `Run From Source` section.
- Confirm files are in:
  - `model_zoo/tflite/`
  - `model_zoo/onnx/`
  - `model_zoo/checkpoints/`

#### `Q: No .wav files found`
- Both offline scripts scan only the exact folder given by `--noisy_dir` (non-recursive).
- Ensure input files use `.wav` extension.

#### `Q: Real-time demo has audio device errors`
- Check microphone permissions and default input/output device settings.
- Install host audio dependencies for `sounddevice` (PortAudio packages on your OS).

#### `Q: Real-time GUI does not open`
- Ensure Qt dependencies from `requirements.txt` installed successfully.
- On headless servers, run offline enhancement instead.

#### `Q: I get import/module errors when running commands`
- Run from repo root and use module form exactly as documented (`python -m ...`).
- Activate your virtual environment before running commands.

#### `Q: CPU is too slow for my target`
- Try smaller models (`baseline`, `dpdfnet2`).
- Benchmark ONNX runtime using `python -m onnx_model.infer_dpdfnet_onnx ...` and compare RTF.

## Evaluation Metrics

To compute *intrusive* and *non-intrusive* metrics on our [DPDFNet EvalSet](https://huggingface.co/datasets/Ceva-IP/DPDFNet_EvalSet), we use the tools listed below. For aggregate quality reporting, we rely on PRISM, the scale‑normalized composite metric introduced in the DPDFNet paper.

### Intrusive metrics: PESQ, STOI, SI-SNR
We provide a dedicated script, `pesq_stoi_sisnr_calc.py`, which computes **PESQ**, **STOI**, and **SI-SNR** for paired *reference* and *enhanced* audio. The script includes a built-in auto-alignment step that corrects small start-time offsets and drift between the reference and the enhanced signals before scoring, to ensure fair comparisons.

### Non-intrusive metrics
- **DNSMOS (P.835 & P.808)** - We use the **official** DNSMOS local inference script from the DNS Challenge repository: [`dnsmos_local.py`](https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos_local.py). Please follow their installation and model download instructions in that project before running. 
- **NISQA v2** - We use the **official** NISQA project: <https://github.com/gabrielmittag/NISQA>. Refer to their README for environment setup, pretrained model weights, and inference commands (*e.g.*, running `nisqa_predict.py` on a folder of WAVs).


## Citation

```bibtex
@article{rika2025dpdfnet,
 title = {DPDFNet: Boosting DeepFilterNet2 via Dual-Path RNN},
 author = {Rika, Daniel and Sapir, Nino and Gus, Ido},
 year = {2025},
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
