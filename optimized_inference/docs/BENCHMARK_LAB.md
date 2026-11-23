# Reproducible CPU benchmark laboratory

The repository-root `justfile` is the mandatory performance-study entry point.
It uses the Python standard library, systemd/cgroup v2, GCC, `perf`, and `just`.
The control backend is validated on the local AMD CPPC machine. It discovers
topology instead of assuming a CPU count or hardcoding L3 siblings. Unsupported
hardware or unverifiable frequency feedback is an error, not a weaker silent
fallback. Commands below run from the repository root.

## Recommended complete session

Finish runtime builds and numerical checks first. Construct a JSON manifest
using the example below and absolute artifact paths. Choose fixed counts,
randomization seed, precision limit and acceptance criteria before collection.

```sh
just bench-test
just bench-session /absolute/path/manifest.json .local/my-study
```

The session compiles the common measurement worker before preparing isolation,
then runs prepare, A/A, comparison, continuous-call diagnostics and separate
hardware profiles. It always restores the environment in a `finally` block,
including on failure or SIGINT/SIGTERM. The output directory must not already
exist; completed studies must not be overwritten or extended opportunistically.
An independent monotonic systemd timer restores settings if the controller is
killed. SIGKILL/power loss cannot be handled by a process `finally`; `/run` state
and the independent timer cover the live-machine case until its recorded TTL.

Reports include `prepare.json`, `aa/result.json`, `compare/result.json`, raw
binary64 observations, plans/hashes, separate perf data, `RESULTS.md`, and
`restore.json`. A failure retains partial evidence and produces an error;
it never prints a successful readiness/restoration verdict without readback.
Controller failure JSON, including the reason and rollback verdict, is retained
in the session error log even when preparation fails before session ownership.
Do not treat a preparation-only retry as a new statistical sample; retain its
failed attempt and never overwrite or recollect a begun timing journal.

## Manual session

```sh
just bench-prepare                  # CPU 3; nominal non-boost frequency; 1800 s TTL
just bench-ready                    # structural checks + three frequency probes
just bench-aa manifest.json .local/control
just bench-compare manifest.json .local/comparison .local/control/result.json
just bench-profile manifest.json .local/hardware
just bench-restore
just bench-status
```

`bench-prepare` accepts CPU, frequency in kHz (or `auto`), and TTL in seconds:
`just bench-prepare 3 auto 1800`. A second active session is refused. The TTL
is 60–7200 seconds and is not silently renewed. `bench-restore` is idempotent;
it terminates remaining workers in this laboratory's cgroup. `bench-exec`
executes an absolute executable as the original ordinary user with zero
capabilities and no-new-privileges. It does not grant privileges to a benchmark.
For arbitrary manual commands the caller must still restore afterward; use the
managed session for automatic cleanup. `bench-status` is read-only; `bench-ready`
also refreshes actual-frequency calibration and restores on calibration failure.

The optional fourth preparation argument selects `steady` (default) or `boost`:
`just bench-prepare 3 auto 1800 boost`. Manifests use `"power_profile": "boost"`.
Both reserve the same L3 domain and park the worker's SMT sibling. `steady`
disables Boost, fixes nominal-frequency bounds and restricts deep worker idle.
`boost` enables Boost, uses dynamic min/max bounds with governor/EPP performance,
and retains original idle settings. This is an isolated Boost experiment, not
the complete normal-desktop/real-audio environment. Boost calibration records
mean/SD and checks permitted bounds with 3% counter tolerance; it explicitly
does not claim constant frequency. Never apply the steady-profile 1%/0.3%
stability gate to variable-frequency Boost or relabel its range check as one.

Preparation compiles the worker first. Do not invoke `bench-tools` during an
active study: changing the probe binary invalidates the saved hash. Runtime
libraries must likewise remain unchanged. Profile recordings and counter jobs
are never mixed into latency estimates.

## Environment contract

- Discover the worker's physical SMT siblings and unified L3 domain.
  Reserve the complete online L3 domain with a cgroup v2 isolated partition;
  admit only the worker CPU into its child cgroup. Keep ordinary system/user
  work on the remaining housekeeping CPUs and verify every live thread's
  effective affinity. Zombies are not runnable and are excluded.
- Stop irqbalance temporarily, steer writable IRQ affinities, default IRQ mask,
  unbound workqueues, movable kernel workers and watchdog placement away from
  the reserved domain. Retain unmovable IRQ/kernel diagnostics. Firmware,
  per-CPU work, shared RAM and package effects cannot be eliminated this way.
- Save every cpufreq policy before disabling Boost, not only the worker policy:
  the boost control can change other policies' limits. Set worker governor/EPP
  to performance and request equal min/max at a non-boost nominal frequency.
  Park the worker's online SMT sibling. Disable idle states with declared exit
  latency above 1 µs on the worker only; leave the rest of the machine's idle
  policy intact. This is a controlled laboratory environment, not a deployment
  recommendation or evidence for normal power-policy performance.
- Verify actual frequency using CPPC delivered/reference counters under three
  600 ms AVX2 load probes. The backend requires positive nominal frequency and
  equal nominal/reference performance units. Default readiness tolerances are
  mean within 1% of target and sample SD <=0.3% of measured mean. Retain every
  probe; actual hardware frequency may differ slightly from the request.
- Readiness additionally verifies policy settings, online CPUs, the independent
  restore timer and absence of ordinary background threads in the reserved L3
  domain. Snapshot worker IRQ/softirq counters before/after timing. No claim of
  an interrupt-free CPU or physically exclusive cache/package is made.

## Transaction and privilege boundaries

Root controls only configuration and perf access. State is root-owned under
`/run/dpdfnet-bench-lab` with mode 0700 and a process lock. Rollback data are
saved and fsynced before writes. An immutable copy of the rollback controller
is installed there before the independent restoration timer is armed. The
timer is scoped to its session ID, so a delayed old timer cannot restore a new
session. Managed cleanup also checks the ID.
The controller and measurement-worker hashes are frozen for the active session;
readiness/launch refuses edits. The validated backend requires the local TSC
clocksource. If the workspace is unavailable, the saved rollback can be invoked
with `sudo python3 /run/dpdfnet-bench-lab/rollback.py restore`.

CPU hotplug, every original frequency policy, Boost, IRQ/workqueue/watchdog and
idle settings are restored with readback; moved live kernel tasks are matched
by PID plus start time to avoid PID-reuse errors. Cgroups and temporary units
are removed and irqbalance returned to its original state. The original root
controller set and CPU online mask are verified. Existing configuration for the
owned systemd unit names is refused rather than overwritten. Restoration errors
remain explicit and retryable; a second successful restore performs no writes.

## Measurement contract

One executable loads each selected library with `dlopen`; no separate benchmark
build/code layout is introduced for the two treatments. Readable ELF pages,
buffers and timing samples are prefaulted before 128 warmup calls. Worker timer
slack is set to 1 ns for the process lifetime. Runtime/model code is unchanged.

- `batch`: CLOCK_MONOTONIC_RAW brackets 16 inferences by default, with no clock
  calls between them. Each retained value is amortized time per inference in
  that group. Its SD/tails are group-mean statistics, not individual-call SD or
  a bound on any one call's latency.
- `continuous`: the same clock brackets each inference, providing separate
  per-call diagnostics. No CPU-time syscall surrounds individual calls.
- `paced`: individual calls at a 10 ms absolute-monotonic cadence; actual calls
  are timed with MONOTONIC_RAW. Warmup is continuous, so this differs from the
  earlier benchmark's paced warmup. Record late completion before cadence reset.

Thread CPU time is read only before/after the measured window (plus one
untimed warmup read); process resource accounting is outside it. Tracepoint
validation on the local machine found exactly three clock_gettime syscalls per
process in both batch and per-call modes, independent of the number of calls.
CLOCK_MONOTONIC/RAW use the local stable-TSC vDSO path. The worker also reports
UID, no-new-privileges, CPU, hardware frequency feedback, faults, context switches
and a final output/full-state checksum. The checksum supplements, not replaces,
the independent every-frame numerical regression tests.

Fresh-process paired blocks are the inference units. Orders are balanced within
each of two inputs; use block counts divisible by four. A/A compares the same
library/binary with itself for every planned ISA/model/cadence. Comparison
requires a matching A/A report with unchanged binary/input hashes. Both phases
retain all observations; a failed A/A precision gate is not repaired by extending
the series. Such comparisons remain diagnostic even if their point estimates
are favorable. Adjust Student-t log-ratio intervals for all planned contrasts.
Report call/group SD separately from process-mean SD, tails and deadline misses.

## Manifest example

```json
{
  "schema_version": 1,
  "cpu": 3,
  "frequency_khz": "auto",
  "ttl_seconds": 1800,
  "worker": "/absolute/repository/.local/bench-lab/worker",
  "inputs": ["/absolute/original.f32", "/absolute/gain.f32"],
  "variants": {
    "native": {
      "baseline": "/absolute/native-baseline/libdpdfnet.so",
      "candidate": "/absolute/native-candidate/libdpdfnet.so"
    },
    "avx2": {
      "baseline": "/absolute/avx2-baseline/libdpdfnet.so",
      "candidate": "/absolute/avx2-candidate/libdpdfnet.so"
    }
  },
  "models": [2, 8],
  "weights": {"2": "/absolute/model2/weights.f32", "8": "/absolute/model8/weights.f32"},
  "seed": 202611105017,
  "calls": {"batch": 512, "continuous": 256, "paced": 100},
  "batch_size": 16,
  "aa_blocks": 4,
  "compare_blocks": 8,
  "aa_noise_limit_percent": 0.5,
  "minimum_useful_gain_percent": 0.3
}
```

With two ISAs/two models this fixed plan uses 64 A/A processes, 128 comparison
processes, eight continuous diagnostic processes, and 40 separate hardware jobs.
The former two phases use eight family-adjusted contrasts each. There are two
counter repetitions per condition/group and one sampling recording per arm;
these profiles provide descriptive diagnostics, not an independent precise
speedup estimate. Do not pool this non-boost study with earlier Boost-enabled
measurements or transfer its gains directly to a real audio deployment.

## Cadence and batching sweeps

The optional `conditions` array replaces the default batch/paced pair. Each
condition has a unique `id`, `worker_mode`, `calls`, `batch_size` and, for paced
conditions, `period_ns`. For example:

```json
[
  {"id":"continuous","worker_mode":"continuous","calls":512,"batch_size":1},
  {"id":"batch16","worker_mode":"batch","calls":512,"batch_size":16},
  {"id":"batch64","worker_mode":"batch","calls":512,"batch_size":64},
  {"id":"paced5","worker_mode":"paced","calls":64,"batch_size":1,"period_ns":5000000},
  {"id":"paced10","worker_mode":"paced","calls":64,"batch_size":1,"period_ns":10000000},
  {"id":"paced20","worker_mode":"paced","calls":64,"batch_size":1,"period_ns":20000000}
]
```

The worker accepts `paced:PERIOD_NS` (1–100 ms); plain `paced` retains 10 ms.
Late completion is relative to the selected period, while `over_10ms` retains
its absolute 10 ms meaning. Model input still represents 10 ms spectral hops;
5/20 ms test cadence is a synthetic wakeup/workload experiment, not a changed
audio sample rate. Continuous and batch modes do not observe paced deadlines.

Use `family_comparisons` to cover the complete planned family across manifests,
never fewer than the contrasts in one manifest. For the six-condition example,
two ISAs/two models/two power profiles require family size 48. Declare separate
manifests and seeds for power profiles; execute them sequentially with complete
restoration between sessions. Shared session order/time-of-day effects prevent
causal attribution of cross-profile differences to one power/idle setting.

Optional `profile_conditions` selects condition IDs for hardware counters;
`profile_groups` selects primary/frontend groups, `profile_repetitions` fixes
their repeats, `profile_calls` fixes shorter profile-only counts, and
`profile_sample:false` omits separate sampling recordings. Profiled time is
always excluded from latency inference. `continuous_diagnostics:false` avoids
redundant diagnostics when continuous already is a primary condition.

## Recovery after aggregation failure

`just bench-resume MANIFEST EXISTING_OUTPUT` supports a fully collected A/A
phase when comparison has not started. It validates the original schedule,
binary/input hashes, every raw sample hash/count, mean and sample SD, then
reconstructs the derived A/A summary. It archives the original preparation and
restoration reports before starting a fresh equivalent environment for the
remaining comparison/profile phases. It never recollects A/A samples or accepts
an incomplete journal, changed manifest, changed binary, or begun comparison.
Record both preparation epochs and the reason for recovery. It is not a way
to extend a completed study after inspecting an unfavorable result.

## Process layout and complete audio callbacks

Preregister `"layout_profile": "fixed"` when measuring small code changes.
The controller applies Linux `ADDR_NO_RANDOMIZE` to each unprivileged worker
before exec, without changing the machine-wide ASLR setting. The worker reports
the effective personality, API address and state-buffer address. Use real
byte-identical library copies at equal-length absolute paths when necessary:
symlinks are resolved by the launcher and do not equalize argument lengths.
Check address distributions after collection. Fixed layout controls one nuisance
variable; it cannot establish performance under every possible layout. Keep
independent randomized-layout evidence separate and never pool these cohorts.
`"layout_profile": "randomized"` is the default and explicitly clears the flag.
The controller also fixes Python's hash seed and numerical-library thread counts.

A condition can override `worker` and its two `inputs`. The standard executable
`tools/bench_lab/audio_worker.py` accepts decoded mono float32 PCM and measures
the documented NumPy FFT -> ready-library inference -> IFFT/overlap-add callback.
Use individual paced conditions, not batch mode, for this worker. It uses the
same warmup, absolute cadence, per-process privilege/layout checks, CPPC feedback
and perf FIFO protocol as the C spectral worker. Add `audio_pipeline.py` and
`examples/dpdfnet.py` to manifest `extra_files`; pin NumPy's reported version.
This is a host-callback measurement, excluding device I/O, resampling, codecs,
audio-server scheduling and service concurrency. It is not a hardware-loopback
or production deployment test. Counter profiles must inherit the same worker
and input override; their durations remain excluded from latency inference.

Before timing, regenerate spectra from the decoded PCM and compare them with
the pinned spectral fixtures. Compare complete synthesized PCM, every spectral
output and recurrent state for all selected libraries and both models. A final
checksum in a timing process is an additional guard, not a substitute for these
checks. State explicitly when regenerated public/stress fixtures do not reproduce
an unavailable historical corpus.

## Practical noninferiority

Declare `noninferiority_margin_percent` before collection (default 0.5%).
`passes_noninferiority` requires a qualified matching A/A control and a
family-adjusted gain-interval lower bound above the negative margin. It means
losses at or beyond that practical margin are excluded by this experiment's
model; it does not prove zero slowdown or useful acceleration. Report it
separately from `passes_acceptance`, deadline/fault checks and numerical gates.
An unqualified A/A cell remains inconclusive even if its point estimate is
positive. Do not increase the sample after inspecting its result.

For multiple machines, require independent readiness and restoration on each.
A host with only one L3 domain cannot satisfy this backend's reserved-L3 plus
separate-housekeeping contract. Ready binaries may still receive low-priority
numerical portability checks there, with service health checked before/during/
afterward, but do not label those runs isolated performance measurements.
