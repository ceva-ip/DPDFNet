# DPDFNet-8 fullband quality evaluation

Completed comparison of original ONNX, selective FP16, selective W8A8 INT8,
and W7A8 on **50 paired EARS-WHAM_v2 clips: 13.04 minutes, six speakers**.
All 250 system/clip evaluations completed with finite scores and no exclusions.

**W7A8 remains a promising speed/quality tradeoff, but is not quality-identical
to INT8.** Its mean PESQ is 0.00624 lower than INT8, while mean native-48-kHz
SI-SNR is 0.00436 dB higher. SIGMOS signal and overall means are slightly higher,
but their paired intervals include zero. FP16 closely matches original ONNX.
This 50-clip screening run does not establish perceptual equivalence or formal
non-inferiority; no perceptual tolerance was specified beforehand.

## Mean quality against clean references

Unweighted means across the same 50 clips; higher is better for every column.
SIGMOS is non-intrusive and scores each waveform without a clean reference.

| System | PESQ-WB (16 kHz) | STOI | SI-SNR (48 kHz), dB | SIGMOS signal (48 kHz) | SIGMOS overall (48 kHz) |
|---|---:|---:|---:|---:|---:|
| Noisy input | 1.21673 | 0.816856 | 5.31746 | 2.88283 | 1.83966 |
| Original ONNX | 2.33959 | 0.915571 | 15.18558 | 3.57470 | 3.12822 |
| Selective FP16 | 2.33961 | 0.915569 | 15.18517 | 3.57455 | 3.12823 |
| Selective INT8 | 2.34098 | 0.915357 | 15.16621 | 3.57232 | 3.13111 |
| W7A8 | 2.33475 | 0.915466 | 15.17057 | 3.58783 | 3.13720 |

W7A8 improves SI-SNR over noisy input by **9.8531 dB** on average. Relative
to original ONNX, its mean differences are −0.00485 PESQ, −0.000105 STOI,
−0.01502 dB SI-SNR, +0.01314 SIGMOS signal, and +0.00898 SIGMOS overall.
The FP16–ONNX differences are +0.000014 PESQ and −0.000416 dB SI-SNR.

### Paired W7A8 minus INT8

Intervals use 10,000 paired bootstrap replicates, seed 20260926. Speaker-cluster
intervals resample six speakers, retaining all their clips. These are descriptive,
unadjusted 95% intervals across multiple metrics; six clusters are a small sample.

| Metric | Mean difference | Clip bootstrap 95% CI | Speaker-cluster 95% CI |
|---|---:|---:|---:|
| PESQ-WB | −0.006239 | [−0.010505, −0.001981] | [−0.013375, −0.000323] |
| STOI | +0.000109 | [−0.000119, +0.000366] | [−0.000188, +0.000455] |
| SI-SNR, dB | +0.004359 | [−0.014930, +0.027015] | [−0.019135, +0.035150] |
| SIGMOS signal | +0.015518 | [−0.004698, +0.036673] | [−0.000774, +0.030109] |
| SIGMOS overall | +0.006093 | [−0.014925, +0.028211] | [−0.003236, +0.015979] |

The small PESQ decline is consistent across these two resampling methods and
occurs on 36/50 clips. The other primary differences do not establish an
improvement or decline. SIGMOS reverb also declines by 0.02470 versus INT8
(clip CI [−0.04362, −0.00654]); this dimension should not be hidden by the
slightly higher overall mean.

### All SIGMOS dimensions

| System | Coloration | Discontinuity | Loudness | Noise | Reverb | Signal | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Noisy | 3.02652 | 3.71340 | 2.85324 | 1.51705 | 4.13816 | 2.88283 | 1.83966 |
| ONNX | 3.55068 | 3.84444 | 3.46279 | 4.48397 | 4.63520 | 3.57470 | 3.12822 |
| FP16 | 3.55079 | 3.84392 | 3.46234 | 4.48410 | 4.63521 | 3.57455 | 3.12823 |
| INT8 | 3.54429 | 3.84914 | 3.46720 | 4.49427 | 4.64992 | 3.57232 | 3.13111 |
| W7A8 | 3.56285 | 3.83692 | 3.45637 | 4.47805 | 4.62522 | 3.58783 | 3.13720 |

### Largest individual W7A8 declines versus INT8

These are different clips, selected after scoring for inspection, not exclusions.

| Metric | Clip | Speaker / recording | Nominal SNR | Difference |
|---|---|---|---:|---:|
| PESQ | 00101 | p102 / emo_amusement_sentences | 0.4 dB | −0.04942 |
| STOI | 00305 | p105 / sentences_03_whisper | 3.6 dB | −0.001289 |
| SI-SNR | 00829 | p102 / emo_cuteness_sentences | 16.3 dB | −0.13307 dB |
| SIGMOS overall | 00749 | p104 / sentences_02_regular | −2.1 dB | −0.17129 |

Saved, aligned audio for these clips is under
`../scratch/fullband/evaluation/audio/{onnx,fp16,int8,w7a8}/{speaker}/{id}.wav`.
The largest overall-score declines deserve listening review before adopting
W7A8 as a default; aggregate scores alone do not establish audibility.

## Coverage and artifacts

The selected subset contains 8 clips each from p102, p103, p104, and p107,
and 9 each from p105 and p106. It includes 19 recording categories, with
regular, whispered, loud, fast, slow, high/low-pitched, emotional, and free-form
speech. Category allocation and clip IDs were fixed before scoring. The
eligible paired test pool at the pinned revision contained 886 clips.

Nominal loudness-based mixture SNR ranges from −2.1 to 16.7 dB (mean 7.2).
Counts in the ranges below 0, 0–5, 5–10, 10–15, and at least 15 dB are
6, 13, 12, 14, and 5. The official recipe includes a 75 Hz highpass on clean
speech and 10 ms ramps. This is the paired test split, not the blind test set.

- [Summary, paired intervals, speaker/SNR groups and ranked declines](../results/fullband_ears_wham_v2.json)
- [Per-clip CSV scores](../results/fullband_ears_wham_v2.csv)
- [Complete clip scores, input/output hashes and run manifest](../results/fullband_ears_wham_v2_clips.json)
- [Download provenance and subset replay validation](../results/fullband_ears_wham_v2_sources.json)

No kernel or production preset was changed in this evaluation. The libraries
are the same baseline and W7A8 artifacts measured in `INT8_RANGE_EXPERIMENTS.md`.

## Protocol fixed before scoring

- Official [EARS-WHAM_v2](https://github.com/sp-uhh/ears_benchmark) generator
  revision `36dc8a88cb2ebf7cc746b51cf2b876bb570bc3e0`; held-out speakers p102–p107.
  Select clips with proportional recording-category quotas and deterministic hash ordering,
  balancing speaker counts before scoring. Download required speech and noise. ZIP CRC
  checks and extracted-file SHA256 hashes record the inputs.
- Run the official generator with default 48 kHz settings and seed 42,
  omitting its training/validation loop. Preserve every test RNG draw and style
  counter by replaying unselected recordings with zero arrays of their exact
  original shapes, obtained from WAV headers. Every source recording with a
  selected cut is processed normally, including its unselected cuts: upstream
  can reuse a preceding cut's noise residual. Only selected official clip IDs
  are saved. The generator's clipping adjustment changes no random draws.
- All four enhancers receive identical mono 48 kHz waveforms. Reset recurrent
  state once per clip; process continuously at 480-sample hops, drain the
  pipeline, then remove the same 2400-sample delay from every enhanced output.
  No per-system alignment optimization, peak normalization, or output clipping.
- Compute zero-mean SI-SNR from all 48 kHz samples using least-squares target
  projection. Report improvements over noisy input and paired system deltas.
- Use the official [SIGMOS V1](https://github.com/microsoft/SIG-Challenge/tree/main/ICASSP2024/sigmos)
  implementation at 48 kHz, revision `bf4525153b6ed998f19d9e79ff1fd00f55dec42b`.
  Model SHA256 `f939dcc1945055a435565b4369e27dafd0f87df3cea4e2ff6eb81225e52cc53b`.
  Report all seven P.804 dimensions; signal/overall are the primary MOS summaries.
  This is an alpha learned estimator, not a listening-test score.
- PESQ-WB is measured on explicit 48→16 kHz polyphase-resampled copies because
  [PESQ does not support 48 kHz](https://github.com/ludlows/PESQ).
  STOI receives 48 kHz input but its [reference algorithm](https://github.com/mpariente/pystoi/blob/master/pystoi/stoi.py)
  internally resamples to 10 kHz. Neither measures the upper fullband
  spectrum. DNSMOS is not substituted for native 48 kHz SIGMOS.
- FP16 is selective FP16 **weight storage with FP32 arithmetic**, matching the
  implemented preset. INT8 and W7A8 use the same selective layer mask (7), with
  FP32 state, normalization, gates, and remaining operations. This is not a
  comparison against a separately converted all-FP16/all-INT8 ONNX graph.
- Save every clip's scores, hashes, and enhanced FLOAT WAVs. Aggregate paired
  differences with both clip and speaker-cluster bootstrap intervals; inspect
  speaker/SNR groups and the largest W7A8 regressions. Six held-out speakers
  limit population-level inference even when the clip count is large.
- Dataset and downloaded third-party sources/models remain in ignored
  `scratch/fullband`. EARS is licensed CC BY-NC 4.0; this is a local research
  evaluation. No dataset audio is added to version control.

## Reproduction

`native/experiments/Dockerfile.fullband` extends the existing quality image with
CPU PyTorch/torchaudio 2.7.1 and pyloudnorm 0.1.1 for the official generator.
The inference/scoring job runs without network access and records exact package
versions and artifact hashes. Parallel workers process independent clips;
these quality runs are not latency benchmarks.

The subset replay was checked against the official generator on a controlled
six-speaker corpus with multiple cuts per recording. All 24 selected clean/noisy
waveforms matched exactly, as did selected CSV rows and the final NumPy RNG
state. SI-SNR passed an analytical 20 dB orthogonal-noise check and scale/offset
invariance checks. The STFT identity round trip differed by at most 4.47e-8,
and an ONNX alignment diagnostic confirmed the fixed 2400-sample lag.
All 200 saved enhanced waveforms were subsequently checked for 48 kHz mono
format, expected length, finite samples, and exact agreement with the scored
PCM SHA256 hashes (`verify_fullband_audio.py`).

Scripts: `download_fullband.py`, `prepare_fullband_subset.py`, and
`fullband_quality.py` in `native/experiments`. Run with the native_inference
directory mounted at `/bench`. Source metadata and official source files are
downloaded to `scratch/fullband` at the pinned revisions above.

From the repository root in PowerShell, with the existing native model/build
artifacts described in `INT8_RANGE_EXPERIMENTS.md` present:

```powershell
$benchRoot = (Resolve-Path native_inference).Path
docker build -t dpdfnet-native-fullband -f native_inference/native/experiments/Dockerfile.fullband native_inference/native/experiments
docker run --rm --mount "type=bind,source=$benchRoot,target=/bench" --entrypoint python dpdfnet-native-fullband /bench/native/experiments/bootstrap_fullband.py
docker run --rm --mount "type=bind,source=$benchRoot,target=/bench" --entrypoint python dpdfnet-native-fullband /bench/native/experiments/prepare_fullband_subset.py --count 50
docker run --rm --network none --mount "type=bind,source=$benchRoot,target=/bench" --entrypoint python dpdfnet-native-fullband /bench/native/experiments/fullband_quality.py --workers 6
docker run --rm --network none --mount "type=bind,source=$benchRoot,target=/bench" --entrypoint python dpdfnet-native-fullband /bench/native/experiments/summarize_fullband.py
```
