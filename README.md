<h1 align="center">DPDFNet: Boosting DeepFilterNet2 via Dual-Path RNN</h1>
<br></br>

<p align="center">
  <a href="https://your-project-url.com">
    <img src="https://img.shields.io/badge/Project-Page-orange" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/your-paper-id">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b" alt="arXiv Paper">
  </a>
  <a href="https://huggingface.co/Ceva-IP/DPDFNet">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow" alt="HuggingFace Models">
  </a>
</p>

<p align="center">
  <sub><em><strong>--- This is the official repository for the DPDFNet paper ---</strong></em></sub>
</p>

## Abstract
We present DPDFNet, a causal single-channel speech enhancement model that extends DeepFilterNet2 architecture with dual-path blocks in the encoder, strengthening long-range temporal and cross-band modeling while preserving the original enhancement framework. In addition, we demonstrate that adding a loss component to mitigate over-attenuation in the enhanced speech, combined with a fine-tuning phase tailored for “always-on” applications, leads to substantial improvements in overall model performance. To compare our proposed architecture with a variety of causal open-source models, we created a new evaluation set comprising long, low-SNR recordings in 12 languages across everyday noise scenarios, better reflecting real-world conditions than commonly used benchmarks. On this evaluation set, DPDFNet delivers superior performance to other causal open-source models, including some that are substantially larger and more computationally demanding. We also propose an holistic metric named PRISM, a composite, scale-normalized aggregate of intrusive and non-intrusive metrics, which demonstrates clear scalability with the number of dual-path blocks. We further demonstrate on-device feasibility by deploying DPDFNet on Ceva-NeuPro™-Nano edge NPUs. Results indicate that DPDFNet-4, our second-largest model, achieves real-time performance on NPN32 and runs even faster on NPN64, confirming that state-of-the-art quality can be sustained within strict embedded power and latency constraints.

---

## Repository Overview

This repo includes:
- **Offline enhancement** for a folder of WAV files (`enhance.py`)
- A **real-time microphone demo** with live spectrograms and A/B playback (`real_time_demo.py`)
- Pre-exported **TFLite models** expected under: `model_zoo/tflite/*.tflite`

Supported model names (TFLite files):

| Model           | Params [M] | MACs [G] | TFLite Size [MB] | Intended Use                    |
| --------------- | :--------: | :------: | :--------------: | ------------------------------- |
| baseline.tflite |    2.31    |   0.36   |        8.5       | Fastest / lowest resource usage |
| dpdfnet2.tflite |    2.49    |   1.35   |       10.7       | Real-time / embedded devices    |
| dpdfnet4.tflite |    2.84    |   2.36   |       12.9       | Balanced performance            |
| dpdfnet8.tflite |    3.54    |   4.37   |       17.2       | Best enhancement quality        |



## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


Both scripts look for models under `./model_zoo/tflite` by default.


## Offline Enhancement

Enhance all `*.wav` files in a folder (**non-recursive**) and write enhanced WAVs to an output folder:

```bash
python enhance.py \
  --noisy_dir /path/to/noisy_wavs \
  --enhanced_dir /path/to/output \
  --model_name dpdfnet8
```

What the script does:
- Loads audio (any sample rate), converts to **mono**, resamples to **16 kHz** for the model.
- Runs the model **frame-by-frame in streaming mode**.
- Reconstructs audio with iSTFT, resamples back to the original sample rate, and saves **mono PCM_16 WAV** outputs.

Output filenames are created as:
```
<original_stem>_<model_name>.wav
```

---

## Real-Time Microphone Demo

The real-time demo performs streaming enhancement on microphone input, displays **Noisy vs Enhanced** live spectrograms, and lets you switch playback between the two.

### Run
```bash
python real_time_demo.py
```

### Configure
Edit constants near the top of `real_time_demo.py`:
- `MODEL_NAME`: `baseline | dpdfnet2 | dpdfnet4 | dpdfnet8`

### Usage
- Speak into your microphone.
- Use the UI buttons to switch playback:
  - **Noisy**: plays raw mic input
  - **Enhanced**: plays model output
- The console prints **ms per frame** to help assess real-time performance.

---

## Citation
If you use this work, please cite the paper:

```bibtex
@article{rika2025dpdfnet,
  title   = {DPDFNet: Boosting DeepFilterNet2 via Dual-Path RNN},
  author  = {Rika, Daniel and Sapir, Nino and Gus, Ido},
  year    = {2025},
}
```

## License

Apache License 2.0. See `LICENSE-2.0.txt`.
