# DPDFNet-2 overview evaluation and fitted-candidate RSS

Measured 2026-09-27 on the Intel i7-8700 in Linux Docker/WSL2. The
[README overview](../README.md) contains the comparison tables for both models.
This extension builds the existing W7A8 + pack32 + fitted degree-5 GRU kernels
against DPDFNet-2's generated model and weights. It does not modify production
kernels or overwrite previously measured libraries.

## Validation and timing

The new `build/w7_followup_pack_fit5_2/libdpdf_full.so` passes all five CTest
contracts: INT8, layout, extended, full and native. A 300-frame recurrent FP32
check against original ONNX also passes: waveform difference SNR 110.77 dB,
maximum absolute spectrum error 1.36e-5, and state error 1.53e-5. That check
validates the generated model with the fitted gates; it does not establish
quantized-output equivalence. Quantized quality is evaluated separately.

All four DPDFNet-2 variants run sequentially at standalone 10 ms cadence, with
100 warmup and 1,000 measured calls per run, four repeats and rotating order.
The table uses the median of four run means. Python wrappers allocate returned
outputs, as in the earlier ONNX comparison. No outliers are removed. No quality
evaluation runs concurrently with the latency benchmark.

The fitted candidate reduces typical latency 12.1% versus selective INT8 and
59.3% versus ONNX in this matched run. All 16,000 measured calls finish their
inference within 10 ms, but this is not a deadline guarantee or a measurement
of scheduling lateness. Its maximum is 7.605 ms, compared with INT8's 1.976 ms.
The reason for that individual spike was not traced in this experiment.

Raw run distributions, parity results and model/library/weight SHA-256 hashes:
[timing results](../results/dpdfnet2_overview_timing.json).

## Memory

The original fresh-process probe is reused without changing its measurement
code. Four independent processes per fitted candidate import the same common
Python/NumPy/ORT dependencies before sampling baseline RSS, create only the
native model, unmap source weights, and warm up for 120 hops before measuring.
The median RSS increment is **7.47 MiB for DPDFNet-8** and **6.03 MiB for
DPDFNet-2**. These exclude the imported runtime baseline and are not total
application memory. The raw record also retains total and peak process RSS.

Native owned allocations are 6,738,951 and 5,406,675 bytes respectively,
unchanged from each model's selective INT8. Small RSS differences between
sessions should not be interpreted as an optimization benefit.

Raw fresh-process samples and artifact hashes:
[RSS results](../results/overview_fitted_rss.json).

## Quality scope

All four DPDFNet-2 variants use the same six mixtures as the DPDFNet-8 overview:
`00033`, `00046`, `00084`, `00133`, `00200`, `00364`, one per held-out EARS
speaker. The existing fullband evaluator runs each file continuously, resets
state between files, and aligns output with the existing 2,400-sample delay.
All inference and waveform processing are at 48 kHz. SI-SNR and SIGMOS use
48 kHz; PESQ-WB explicitly resamples to 16 kHz and STOI internally uses 10 kHz.
Means are unweighted across the six clips. This is a small engineering screen,
not a full 50-clip DPDFNet-2 evaluation or proof of perceptual equivalence.

The [quality results](../results/dpdfnet2_overview_quality.json) preserve every
clip's metrics, input/output hashes, model/library hashes and SIGMOS model hash.
Output WAVs are under `scratch/fullband/overview_dpdfnet2/audio/`.

## Reproduction

Use the existing offline development/fullband images, mounting
`native_inference` at `/bench` with working directory `/bench`. The prepared
`scratch/w7_followup/pack_fit5` source tree from the
[W7 follow-up](W7_LATENCY_FOLLOWUP.md#artifacts-and-reproduction), generated models and weights,
and the existing six EARS mixtures/SIGMOS assets must be present.

Run the stages sequentially; use the development image for build, memory and
timing, and the fullband image for quality. Set `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1` for timing and quality.

```sh
python native/experiments/overview_models.py build
python native/experiments/overview_models.py memory
python native/experiments/overview_models.py timing
python native/experiments/overview_models.py quality
```

The [driver](experiments/overview_models.py) reuses the existing memory probe,
timing harness and fullband scorer. Quality results resume per clip, rejecting
cached results when the driver or model artifacts change.
