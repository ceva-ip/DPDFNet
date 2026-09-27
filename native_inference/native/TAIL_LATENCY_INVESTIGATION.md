# Why a faster model can have a worse observed peak

2026-09-27 follow-up to [W7A8 latency optimization](W7_LATENCY_FOLLOWUP.md).

**The mean improvement is repeatable; a lower maximum is not established.**
The optimization saved about 0.10 ms of normal inference work. The maximum is
the slowest complete call seen in a finite run, including wrapper work and
variations in execution conditions. One unrelated millisecond-scale event can
outweigh that saving. An observed maximum is not a worst-case execution bound.

This investigation removes an observed allocation/GC source of pauses, but
**does not demonstrate that all peak latency is fixed**. Occasional slow calls
remain in a C-only loop. CPU affinity alone did not consistently help.

## What the previous measurements actually establish

- The fitted candidate's two standalone runs had maxima of **3.977 and
  2.462 ms**, despite identical model code and input sequences. Reference W7A8
  had **3.035 and 2.562 ms**. One run dominates the reported maximum.
- An earlier allocating-wrapper diagnostic recorded a **9.406 ms W7A8 call
  containing 7.156 ms of GC activity**. That directly identifies a real source
  of large pauses in the Python benchmark/integration path.
- That GC example does **not** identify the cause of the fitted candidate's
  separate 3.977 ms call. The old cadence probe kept per-call details only above
  10 ms. It did not record GC or CPU identity for that call.
- The affected standalone run recorded no Linux guest context switches, and
  thread-CPU time rose with wall time. It would therefore be incorrect to claim
  that ordinary guest descheduling explains the specific old maximum.

The allocating wrapper creates a new output, a new 90,228-float state array,
and native pointer objects on every hop. Automatic cyclic GC is triggered by
object allocation activity; its start/stop callbacks support the measurements
used here. See the [Python 3.11 GC documentation](https://docs.python.org/3.11/library/gc.html).

## New controlled experiment

Five execution conditions, both preserved W7A8 and fitted W7A8, two reversed-order
repetitions, 1,500 timed hops per run, plus 100 warmup hops: **30,000 timed calls**.
Every run uses standalone 10 ms cadence. Inputs and initial states are identical.
No build or quality job ran concurrently with this benchmark.

Each call records wall time, calling-thread CPU time, associated GC duration/count, page
faults, guest context switches, CPU identity before/after the call, wake lateness,
and completion time relative to the scheduled hop. Full traces are retained.
GC remains enabled. Initialization garbage is collected before each warmup,
equally for all conditions; this is not a reproduction of every prior GC history.
In this 30,000-call dataset, the GC association window includes the short
post-call CPU/rusage bookkeeping interval, while the wall-time interval ends
before that bookkeeping. GC counts therefore identify nearby activity, not
necessarily a pause entirely inside the timed function. The probe was subsequently
tightened to snapshot GC counters immediately at the end of timing. Zero events
in the wider original window also means zero events inside the timed call.

The pinned condition restricts only the test thread to guest CPU 6 and restores
its affinity afterward. Affinity limits where a thread may execute; it does not
reserve that CPU exclusively. See [Linux `sched_setaffinity`](https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html).
Inside WSL2, this is a guest CPU, not exclusive ownership of a physical Windows
core. WSL2 uses a managed VM, as described by [Microsoft](https://learn.microsoft.com/en-us/windows/wsl/compare-versions).

Pooled results below use **3,000 samples per cell**, rather than averaging run
percentiles. Units are milliseconds. Maxima remain noisy; the table is not a
ranking based solely on the single fastest or slowest observation.

| Execution condition | W7A8 mean / p99 / max | Fitted mean / p99 / max |
| --- | ---: | ---: |
| Allocating Python wrapper, free affinity | 1.978 / 2.413 / 3.475 | 1.855 / 2.285 / 3.064 |
| Reusable buffers, free affinity | 1.968 / 2.389 / 3.521 | 1.863 / 2.451 / 3.503 |
| Reusable buffers, pinned | 1.973 / 2.475 / 3.239 | 1.833 / 2.369 / 3.485 |
| C-only loop, free affinity | 1.959 / 2.463 / 3.330 | 1.830 / 2.245 / 3.301 |
| C-only loop, pinned | 1.942 / 2.423 / 3.441 | 1.855 / 2.369 / 3.459 |

All 30,000 model calls completed before their next scheduled 10 ms hop, including
observed wake lateness. Worst completion was **3.672 ms after scheduled release**.
This excludes FFT, audio-device work and resampling; it is not a complete audio
pipeline deadline guarantee.

## What improved, and what remains unexplained

**Buffer reuse removed the observed allocation-related events.** Each model had
30 calls associated with GC activity in its 3,000 allocating-wrapper samples. Page faults
occurred in 1 W7A8 call and 11 fitted calls. Reusable-buffer and C-only conditions
had **zero GC events and zero page faults inside their measured calls**.
The new run did not reproduce the earlier 7.156 ms GC pause: its observed GC
events were much shorter. The earlier event remains evidence of a possible
pause source, not an event silently removed from the new results.

**Removing Python did not eliminate the remaining tail.** In the C-only pinned
fitted run, the slowest call took 3.459 ms wall time and 3.452 ms thread-CPU time.
It had no observed GC, page fault, CPU migration or guest context switch.
The equivalent maxima in its two repetitions were 2.807 and 3.459 ms. This is
why pinning or disabling GC cannot honestly be presented as a complete fix.

**The extreme frames were generally not repeatable.** Replaying identical input
frames and reset state gave only 0–1 common frames among each run's 15 slowest
frames, across all ten condition/model combinations. For C-only pinned fitted
inference, the overlap was zero and same-frame latency correlation was 0.036.
This is evidence against a deterministic input-specific worst path being the
dominant explanation for these peaks; it is not a proof that input cost never
varies.

CPU frequency changes, cache/memory contention, interrupts and host/VM effects
are plausible remaining contributors. We have not separated them. Thread-CPU
time measures time, not instructions or clock frequency, and guest counters do
not expose every host event. A read-only attempt to open a user-mode hardware
cycle counter returned `EPERM` in the existing container. No capabilities,
scheduling priorities, power settings or host security settings were changed.

## Implemented fix and how to use it

[StreamingRunner](streaming_runner.py) allocates the spectral input, output and
state buffers once, caches the native pointers, and reuses state in place. It
avoids per-hop large-array allocation and pointer reconstruction. The existing
ONNX-compatible allocating wrapper is preserved so its returned-output ownership
and historical benchmarks do not change.

```python
from streaming_runner import StreamingRunner

# model is an initialized ExtendedModel for the chosen research build.
stream = StreamingRunner(model)
for spectrum in spectra_source:  # float32, shape (1, 1, 481, 2)
    enhanced = stream.process(spectrum)
    consume_now(enhanced)
stream.reset()                  # before starting an independent stream
```

`enhanced` and `stream.state` are borrowed buffers, overwritten by the next call.
Copy explicitly when retaining history. Use one runner/model context per stream
and thread. Do not close the model while the runner is processing.

Validation checks each model's allocating and reusable paths for **256 frames
of byte-identical output/state**, reset behavior, and invalid input rejection.
The C-only loop also matches the final output/state after 128 frames. Calling a
runner after its model is closed is rejected. No weights, activation formulas or
production kernels changed in this follow-up.

The [existing C ABI](integration/README.md) already supports caller-owned buffers
and in-place state. A native consumer using it correctly does not have this
Python GC path to fix; the new Python runner makes the research integration use
the same buffer-lifetime approach.

## Recommendation

1. Use persistent buffers and initialized/warmed model state in the streaming
   path. Keep loading, allocation, logging and large cleanup work outside it.
2. Keep the faster fitted kernel as a candidate, subject to its documented
   quality coverage. These results do not justify reverting it solely because
   one prior maximum was higher.
3. Do not enable affinity as an unconditional fix: it did not reliably lower
   peaks here. Test affinity/isolation and audio-thread scheduling on the actual
   deployment host with its real workload.
4. For further peak reduction, collect host-native CPU/cache/frequency and
   scheduler traces, then target the observed cause. Do not claim that disabling
   Python GC globally or changing process priority will remove every spike.

The allocation-related pause source is addressed. A consistently lower absolute
peak, or a hard worst-case bound, has **not** been established in this shared
WSL2 environment. Judge subsequent changes by repeated p99/p99.9, maxima,
completion deadlines and realistic load together.

## Reproduction and evidence

From `native_inference` in the existing development container:

```sh
python native/experiments/tail_latency_probe.py
python native/experiments/analyze_tail_latency.py
```

- [Per-run and pooled results](../results/w7_tail_latency.json)
- [Raw per-call traces](../results/w7_tail_latency.npz)
- [Repeated-frame and event analysis](../results/w7_tail_analysis.json)
- [Exact probe source revision for the 30,000-call dataset](../results/w7_tail_probe_source_v1.txt)
  and [400-call smoke check of the tightened GC boundary](../results/w7_tail_probe_smoke.json)
- [Python probe](experiments/tail_latency_probe.py), [C-only loop](experiments/tail_loop.c),
  and [analysis script](experiments/analyze_tail_latency.py)
- Prior [wrapper GC records](../results/w7_followup_pack_fit5_wrapper.json) and
  [standalone cadence results](../results/w7_followup_pack_fit5_final.json)
