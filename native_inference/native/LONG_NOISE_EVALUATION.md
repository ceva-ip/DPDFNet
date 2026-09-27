# DPDFNet-8: continuous two-minute noise-only tests

Completed three **120-second, mono 48 kHz synthetic noise files** through
original ONNX, selective FP16, selective INT8, and W7A8: twelve runs total.
All output and recurrent-state values remained finite over all **144,072 model
hops**, including flush hops. No output clipped, and no 20 ms output window
had greater RMS than the corresponding input window.

**W7A8 showed no sustained-run instability on these fixtures.** Suppression
remained strong through the last 30 seconds. These are controlled synthetic
noise tests, not evidence covering every real-world noise or speech-like sound.

## Inputs and method

- White noise: seeded Gaussian broadband noise.
- Pink noise: random noise shaped approximately as 1/f power above 20 Hz,
  with a flat shaping floor below 20 Hz and zero DC.
- Mechanical-style noise: colored hiss, 60/120 Hz hum, a varying 17-second
  amplitude envelope, and 80 irregular decaying noise bursts. This is synthesized,
  not a recording of a specific machine.
- Each file is exactly 5,760,000 samples, normalized to **−25 dBFS RMS**,
  with input peaks checked below full scale. Seed: 20260927. There is no speech.
- Same ONNX model, native libraries, weights, and selective precision settings
  as the preceding [EARS evaluation](FULLBAND_EVALUATION.md).
- Recurrent state is initialized once per file and carried across all 120 seconds.
  No chunk resets. Process 480-sample hops, drain the pipeline, and remove the
  same 2400-sample delay from all outputs. Save aligned 120-second FLOAT WAVs.
- Check output/state finiteness on every hop, state magnitudes every ten seconds,
  per-second output levels, 20 ms RMS gains, and several longer time segments.
- PESQ, STOI, and SI-SNR against silent speech references are not meaningful
  here. SIGMOS is a speech-quality estimator and is not used to score pure noise.
  Attenuation is input RMS dBFS minus output RMS dBFS; higher means less residual
  noise. These parallel runs are **not latency benchmarks**.

## Whole-file attenuation

All values include startup and the complete aligned 120-second output.

| Noise | ONNX | FP16 | INT8 | W7A8 |
|---|---:|---:|---:|---:|
| White | 88.66 dB | 88.67 dB | 88.78 dB | 89.74 dB |
| Pink | 92.65 dB | 92.65 dB | 91.95 dB | 91.80 dB |
| Mechanical-style | 93.82 dB | 93.81 dB | 93.52 dB | 92.71 dB |

W7A8 versus INT8 is +0.96 dB for white, −0.15 dB for pink, and −0.81 dB
for mechanical-style noise. These differences concern extremely small residuals;
they should not be presented as established perceptual improvements or losses.

## W7A8 output levels and continuity

| Noise | Output RMS | Output peak | Attenuation at 1–30 s | Attenuation at 90–120 s |
|---|---:|---:|---:|---:|
| White | −114.74 dBFS | −81.29 dBFS | 88.98 dB | 90.53 dB |
| Pink | −116.80 dBFS | −71.85 dBFS | 87.78 dB | 98.31 dB |
| Mechanical-style | −117.71 dBFS | −75.01 dBFS | 87.88 dB | 102.32 dB |

All four variants also had stronger attenuation in the last segment than in
the 1–30 second segment. Mechanical noise changes over time, so its segment
differences should not be interpreted solely as model adaptation.

Across all twelve runs, the highest output peak was approximately **−71.85 dBFS**.
The least attenuated 20 ms window still had approximately **57.80 dB** of RMS
attenuation. Every run had zero clipped samples and zero windows amplified
by more than 3 dB (indeed, the maximum window gain was negative in every run).
This checks numerical stability and residual levels, not a listening-based
classification of the remaining artifacts.

## Verification and artifacts

Every saved output was read back and checked for the expected rate and exact
agreement with the scored PCM SHA256. Inputs and outputs contain exactly
120 seconds after common delay compensation. The full result retains input
hashes, library/model hashes, per-second levels, segment levels, state checkpoints,
and the timestamp of each run's worst 20 ms gain.

- [Complete results and provenance](../results/fullband_long_noise.json)
- [Runner](experiments/long_noise_quality.py)
- Inputs: `../scratch/fullband/long_noise/{white,pink,mechanical}.wav`
- Outputs: `../scratch/fullband/long_noise/{noise}_{onnx,fp16,int8,w7a8}.wav`

Using the existing benchmark image and model artifacts, from the repository root:

```powershell
$benchRoot = (Resolve-Path native_inference).Path
docker run --rm --network none --mount "type=bind,source=$benchRoot,target=/bench" --entrypoint python dpdfnet-native-fullband /bench/native/experiments/long_noise_quality.py
```
