# DPDFNet-8: low-level and clean-speech robustness

Completed **10 low-level EARS mixtures and 2 clean references**, comparing
original ONNX, selective FP16, selective INT8, and W7A8. All 60 input/system
score sets completed without metric errors. Saved PCM hashes, sample counts,
finite samples, and mono 48 kHz format were verified for every scored waveform.

**W7A8 behaves similarly to the existing model, but the low-level test exposes
a shared weakness on one clip.** Nine clips have relatively small SI-SNR changes
after level reduction. On clip 00525, all four implementations lose about 6 dB
and attenuate additional speech-active frames. This behavior is already present
in original ONNX; it is not specific to W7A8.

## Definition and controls

- Low level means **clean-reference full-clip RMS = −50 dBFS**, not peak level,
  LUFS, microphone SPL, or −50 dB SNR. Each noisy mixture receives the same
  scalar as its clean reference, preserving its original speech/noise ratio.
  This tests absolute-level robustness, not speech reduction against fixed noise.
- Ten clips were selected from the previous 50 using fixed SHA256 ordering and
  speaker balancing, without consulting scores. All six speakers are represented.
  Original clean RMS levels span −47.59 to −31.84 dBFS, so every selected input
  was attenuated. The selection includes regular, slow, whispered, emotional,
  and free-form speech; nominal mixture SNR spans −0.7 to 15.6 dB.
- Clean controls use the noise-free paired references for p106
  `sentences_22_regular` (00033) and p107 `freeform_speech_01` (00133), at their
  existing levels of −41.06 and −38.58 dBFS RMS. The EARS-WHAM reference
  preparation includes its 75 Hz highpass and ramps. No noise is added.
- Same model/library hashes, precision configurations, recurrent reset, 480-sample
  hops, and fixed 2400-sample output delay correction as the
  [50-clip comparison](FULLBAND_EVALUATION.md). No input AGC or model changes.
- Inference, SI-SNR, and SIGMOS use 48 kHz. PESQ-WB requires 16 kHz copies;
  STOI internally resamples to 10 kHz. FP16 is selective FP16 weight storage
  with FP32 arithmetic, matching the existing preset.
- Raw SIGMOS receives the actual quiet output. A second diagnostic applies the
  inverse input gain **after enhancement**, restoring the original playback
  level equally for all outputs. This helps separate level effects in SIGMOS
  from processing artifacts. It is not a model-side normalization workaround.

## Low level: means over all 10 clips

No clips, including the outlier, are excluded from these means.

| System | PESQ-WB | STOI | SI-SNR at 48 kHz, dB | Raw SIGMOS overall | Level-restored SIGMOS overall |
|---|---:|---:|---:|---:|---:|
| Noisy input | 1.32662 | 0.882852 | 8.93049 | 2.32936 | 2.01264 |
| ONNX | 2.51381 | 0.943407 | 15.84397 | 3.01816 | 3.29303 |
| FP16 | 2.51379 | 0.943402 | 15.84340 | 3.01828 | 3.29328 |
| INT8 | 2.50180 | 0.943114 | 15.81084 | 3.01456 | 3.29062 |
| W7A8 | 2.50807 | 0.943069 | 15.82981 | 3.03654 | 3.29377 |

At −50 dBFS, W7A8 differs from ONNX by −0.01416 dB SI-SNR and −0.00574
PESQ; versus INT8 it is +0.01896 dB and +0.00627 PESQ. These are descriptive
results from ten clips, not a formal equivalence test.

Paired change from the **same ten clips at their original levels**:

| System | Mean SI-SNR change, dB | Mean PESQ change after level restoration | Mean restored SIGMOS overall change |
|---|---:|---:|---:|
| ONNX | −0.60095 | −0.00515 | −0.02238 |
| FP16 | −0.60039 | −0.00507 | −0.02193 |
| INT8 | −0.60993 | −0.01949 | −0.01609 |
| W7A8 | −0.61939 | −0.01113 | −0.03332 |

Mean clean-speech projection gain is −0.512 dB for ONNX, −0.512 dB for
FP16, −0.502 dB for INT8, and −0.489 dB for W7A8. This least-squares gain
diagnostic complements SI-SNR, which is insensitive to a uniform gain change.

### Low-level outlier: 00525, p103, emo_desire_sentences

Original speech RMS is −37.10 dBFS; speech and noise were both attenuated by
12.90 dB. Nominal mixture SNR remains 9.2 dB.

| System | SI-SNR change from original level | Speech-active frames attenuated >20 dB: original → low |
|---|---:|---:|
| ONNX | −5.85845 dB | 4.18% → 9.32% |
| FP16 | −5.85249 dB | 4.18% → 9.32% |
| INT8 | −5.93366 dB | 4.18% → 9.65% |
| W7A8 | −6.06164 dB | 4.18% → 9.32% |

ONNX and W7A8 both flag the 20 ms frames covering **1.10–1.68 seconds**.
INT8 additionally flags 1.08–1.10 seconds. A frame is classified speech-active
when clean RMS is within 20 dB of the clip's maximum frame RMS; a suppression
flag means output RMS is more than 20 dB below that clean frame. This is an
energy diagnostic, not an audited perceptual transcription or listening verdict.

At the low level, ONNX and W7A8 have almost identical SI-SNR on this outlier:
7.61400 versus 7.61533 dB. Their different changes partly reflect different
normal-level scores. The other nine W7A8 changes range from −0.16138 to
+0.15267 dB. This makes the shared base-model level sensitivity the main
follow-up target, rather than a W7A8-specific failure.

## Clean speech: means over 2 clips

| System | PESQ-WB | STOI | SI-SNR at 48 kHz, dB | SIGMOS overall | Speech projection gain |
|---|---:|---:|---:|---:|---:|
| Clean input | 4.64389 | 1.000000 | ∞ (identical reference) | 3.48204 | 0 dB |
| ONNX | 4.03575 | 0.998714 | 29.64029 | 3.41978 | −0.153 dB |
| FP16 | 4.03577 | 0.998717 | 29.63895 | 3.41992 | −0.153 dB |
| INT8 | 4.03582 | 0.998776 | 29.47940 | 3.42180 | −0.123 dB |
| W7A8 | 4.04013 | 0.998772 | 29.22639 | 3.40799 | −0.186 dB |

The enhancer changes clean speech even in original ONNX; this is not a perfect
bypass. W7A8 adds a small SI-SNR reduction: −0.25301 dB versus INT8 and
−0.41391 dB versus ONNX, with nearly identical STOI and PESQ. With only two
signals, the slightly higher W7A8 PESQ should not be interpreted as an improvement.

| Clean clip | ONNX SI-SNR | INT8 SI-SNR | W7A8 SI-SNR | W7A8 PESQ |
|---|---:|---:|---:|---:|
| 00033, p106, regular sentence | 25.81002 | 25.62551 | 25.61168 | 4.24528 |
| 00133, p107, free-form | 33.47057 | 33.33329 | 32.84110 | 3.83499 |

One 20 ms frame at 0.38 seconds in clean clip 00033 meets the suppression
threshold in **all four** variants. No frames meet it in the other clean clip.
No numerical failures, nonfinite output/state, or quantization-specific collapse
were observed; these findings do not imply zero speech alteration.

## Artifacts and reproduction

- [All cases, metrics, levels, comparisons, selection and provenance](../results/fullband_robustness.json)
- [PCM verification and per-case suppression/level-sensitivity audit](../results/fullband_robustness_audit.json)
- Audio: `../scratch/fullband/robustness/audio/{case_id}/{input,clean,onnx,fp16,int8,w7a8}.wav`.
  Low-level audio is saved at its actual test level; no listening gain is baked in.
- Runner: `experiments/robustness_quality.py`; audit: `experiments/audit_robustness.py`.

Using the cached data, models, builds, and Docker image from the preceding run:

```powershell
$benchRoot = (Resolve-Path native_inference).Path
docker run --rm --network none --mount "type=bind,source=$benchRoot,target=/bench" --entrypoint python dpdfnet-native-fullband /bench/native/experiments/robustness_quality.py
docker run --rm --network none --mount "type=bind,source=$benchRoot,target=/bench" --entrypoint python dpdfnet-native-fullband /bench/native/experiments/audit_robustness.py
```
