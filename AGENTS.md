# DPDFNet engineering instructions

Reply to the user in Russian. Keep technical documentation in English.
The optimized Linux x86-64 runtime is in `optimized_inference/`.

## Numerical changes

- Preserve exact quantization, FP operation order, model weights and ABI.
  Do not introduce approximate reciprocals, fast-math or implicit FMA.
- Keep an independent pre-change oracle when refactoring numerical helpers.
- Check both models and AVX2/VNNI, guarded memory, rounding modes, FTZ/DAZ,
  sanitizer contracts, ISA restrictions and stream/state/reset equivalence.
- Keep experimental candidates separate from accepted runtime defaults.

## Performance measurements

- Mandatory entry point: read `optimized_inference/docs/BENCHMARK_LAB.md`, run
  `just bench-test`, and use `just bench-session MANIFEST OUTPUT` for performance
  studies. For manual sessions use `just bench-prepare`, `just bench-ready`,
  `just bench-aa`, then `just bench-compare`, and always `just bench-restore`.
  Do not replace this with ad-hoc taskset/IRQ/power scripts. Unsupported
  hardware must fail clearly; extend and test the controller before proceeding.
- Reserve the complete discovered L3 domain from ordinary background work,
  with housekeeping on another domain. Use the verified `steady` non-boost
  frequency/idle policy as the laboratory baseline. Explicitly preregister
  `boost` when testing dynamic frequency and original idle states; its readiness
  checks frequency bounds, not stability. Readiness requires hardware-counter
  calibration, not just governor/min/max readback. Record requested/measured
  frequency and never pool or transfer gains across power profiles.
- Require an unchanged-binary A/A report before comparing candidates. Retain
  failed precision gates as diagnostic results without optional sample
  extensions. A quiet configuration does not itself prove adequate precision.
- Use the common library-loading measurement worker. Its CPU-time system
  calls are outside measured series; batch observations are amortized means,
  not individual-call latency. Keep per-call continuous/paced diagnostics
  separate. Compile everything before preparation, freeze hashes and do not
  rebuild or tune code/settings during the measured series.
- Finish builds and correctness tests before timing. Reserve one physical
  core together with its SMT sibling using an isolated CPU partition; run
  on one logical CPU and keep its sibling idle/offline. Affinity alone does
  not exclude other work. Verify actual topology and CPU ownership first.
- Save original settings, arm an independent restoration watchdog and use
  try/finally restoration. Verify CPU online state, IRQ/workqueue masks,
  power policy, services and cgroups after completion or failure.
- Steer movable IRQs/workqueues away and record remaining kernel activity.
  Do not promise zero interruptions or exclusive shared-cache/package use.
- Run benchmarks without privileges after setting up isolation. Separate
  unprofiled latency from perf counters/sampling; normalize warmed counter
  windows per call and check event support and multiplexing.
- Fix input hashes, warmup, cadence, process counts, balanced/randomized
  order, A/A controls and acceptance criteria before collecting results.
  For cadence/batch sweeps, declare explicit condition IDs, per-condition sample
  units and counts, and one multiplicity family covering all compared profiles.
  Use a small fixed discovery study and independent confirmation. Respect
  requests for short studies; report limited precision instead of extending
  the sample until a desirable result appears.
- Preregister process layout (`fixed` or `randomized`), verify effective ASLR
  policy and address distributions, and retain the cohorts separately. Use
  equal-length real library paths for fixed-layout comparisons. For full-audio
  claims, test the decoded-PCM FFT/inference/IFFT callback and verify its spectra,
  state and synthesized PCM before timing; spectral API timing alone is narrower.
- Preregister a practical noninferiority margin separately from useful gain.
  Noninferiority is not proof of zero slowdown or superiority. Hosts without a
  separate housekeeping L3 can receive numerical checks, not claims of strict
  isolated performance under this laboratory contract.
- Retain raw observations, slow calls, binary/PGO hashes and failed attempts.
  Preserve controller failure reasons and rollback verdicts in session logs.
  A preparation-only retry must retain its failed audit and must not overwrite
  a begun timing journal or change the registered plan.
  Report mean and sample SD for calls and for process means separately,
  paired process/block confidence intervals, tails and deadline misses.
  Serially dependent calls are not independent statistical replicates.
- Compare incremental changes with the current selected implementation.
  Never add overlapping gains or infer full-graph speed from a microbenchmark.
  Assess continuous and paced workloads separately; record tested CPU scope.
- Document methods, numerical results, uncertainty and default/experimental
  selection. Update affected architecture/build docs and the private backlog.

## Completion

Run appropriate checks and `git diff --check`. Do not push or deploy unless
requested. Include `Documentation:` with updated/not-needed entries in the
final response, alongside checks, measured effects and remaining limitations.
