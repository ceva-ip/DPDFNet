#!/usr/bin/env python3
"""Paired fresh-process comparisons. Blocks, not hops, are statistical units.

Each block is ABBA or BAAB, with the same model/input and two processes per
build. Order is balanced and randomized before collection. Report every run;
never remove outliers. Bootstrap entire blocks, retaining paired observations.
Screening and independent confirmation should use different seeds/output files.
"""
import argparse
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None,
            "median": statistics.median(values), "min": min(values), "max": max(values)}


def student_t_cdf(x, degrees):
    """Integrate the normalized t density with adaptive Simpson quadrature.

    Only used for small research summaries, never in the measured workload.
    Checked against the df=1 analytic CDF and NIST tabulated critical values.
    """
    if x < 0: return 1 - student_t_cdf(-x, degrees)
    coefficient = math.exp(math.lgamma((degrees + 1) / 2) - math.lgamma(degrees / 2)) / math.sqrt(degrees * math.pi)
    def density(v): return coefficient * (1 + v * v / degrees) ** (-(degrees + 1) / 2)
    def integrate(a, b, fa, fm, fb, previous, tolerance, depth):
        middle = (a + b) / 2
        fl, fr = density((a + middle) / 2), density((middle + b) / 2)
        left = (middle - a) * (fa + 4 * fl + fm) / 6
        right = (b - middle) * (fm + 4 * fr + fb) / 6
        delta = left + right - previous
        if depth == 0 or abs(delta) <= 15 * tolerance:
            return left + right + delta / 15
        return (integrate(a, middle, fa, fl, fm, left, tolerance / 2, depth - 1) +
                integrate(middle, b, fm, fr, fb, right, tolerance / 2, depth - 1))
    a, b, c = density(0), density(x / 2), density(x)
    return .5 + integrate(0, x, a, b, c, x * (a + 4 * b + c) / 6, 1e-12, 24)


def student_t_quantile(probability, degrees):
    if not .5 < probability < 1 or degrees < 1: raise ValueError("Invalid t quantile")
    low, high = 0., 1.
    while student_t_cdf(high, degrees) < probability: high *= 2
    for _ in range(50):
        middle = (low + high) / 2
        if student_t_cdf(middle, degrees) < probability: low = middle
        else: high = middle
    return (low + high) / 2


def summarize(rows, seed=20261026, resamples=20000, alpha=0.05):
    groups = {}
    for row in rows:
        groups.setdefault(row["block"], []).append(row)
    effects, paired_cpu, fixed_contrasts, orders = [], [], [], []
    for block in groups.values():
        a = [r for r in block if r["variant"] == "baseline"]
        b = [r for r in block if r["variant"] == "candidate"]
        if len(a) != 2 or len(b) != 2:
            continue
        effects.append(math.log(statistics.mean(r["mean_ms"] for r in b) /
                                statistics.mean(r["mean_ms"] for r in a)))
        paired_cpu.append(math.log(statistics.mean(r["cpu_ms"] for r in b) /
                                   statistics.mean(r["cpu_ms"] for r in a)))
        order = block[0].get("order")
        orders.append(order)
        fixed_contrasts.append(effects[-1] if order == "ABBA" else -effects[-1])
    if len(effects) < 2:
        raise ValueError("Need at least two complete comparison blocks")
    rng = random.Random(seed)
    n = len(effects)
    boot = sorted(statistics.mean(rng.choices(effects, k=n)) for _ in range(resamples))
    low = boot[max(0, int(resamples * alpha / 2))]
    high = boot[min(resamples - 1, int(resamples * (1 - alpha / 2)))]
    observed = abs(statistics.mean(effects))
    # Respect the balanced randomized assignment. A fixed contrast always means
    # center positions versus outer positions, independently of treatment label.
    positive = orders.count("ABBA")
    if all(o in ("ABBA", "BAAB") for o in orders):
        def randomized(selected):
            return abs((2 * sum(fixed_contrasts[i] for i in selected) - sum(fixed_contrasts)) / n)
        assignments = math.comb(n, positive)
        if assignments <= 20000:
            perm = [randomized(indices) for indices in itertools.combinations(range(n), positive)]
            p = sum(v >= observed - 1e-15 for v in perm) / len(perm)
        else:
            p = (1 + sum(randomized(rng.sample(range(n), positive)) >= observed - 1e-15
                         for _ in range(resamples))) / (resamples + 1)
    elif n <= 16:
        perm = [abs(sum(v if mask & (1 << i) else -v for i, v in enumerate(effects)) / n)
                for mask in range(1 << n)]
        p = sum(v >= observed - 1e-15 for v in perm) / len(perm)
    else:
        p = (1 + sum(abs(sum(v if rng.getrandbits(1) else -v for v in effects) / n)
                     >= observed - 1e-15 for _ in range(resamples))) / (resamples + 1)
    center = statistics.mean(effects)
    radius = student_t_quantile(1 - alpha / 2, n - 1) * statistics.stdev(effects) / math.sqrt(n)
    answer = {"complete_blocks": n,
              "time_reduction_pct": 100 * (1 - math.exp(statistics.mean(effects))),
              "time_reduction_ci_pct": [100 * (1 - math.exp(center + radius)), 100 * (1 - math.exp(center - radius))],
              "ci_method": "Student t interval for mean paired block log ratio, df=n-1",
              "bootstrap_time_reduction_ci_pct": [100 * (1 - math.exp(high)), 100 * (1 - math.exp(low))],
              "ci_level": 1 - alpha, "bootstrap_resamples": resamples,
              "two_sided_sign_randomization_p": p,
              "randomization_method": "Balanced ABBA/BAAB assignment" if all(o in ("ABBA", "BAAB") for o in orders) else "Unrestricted sign randomization",
              "cpu_time_reduction_pct": 100 * (1 - math.exp(statistics.mean(paired_cpu))),
              "block_log_ratios": effects,
              "statistical_unit": "ABBA/BAAB block; individual hops are correlated"}
    for variant in ("baseline", "candidate"):
        runs = [r for r in rows if r["variant"] == variant]
        answer[variant] = {"run_means_ms": distribution([r["mean_ms"] for r in runs]),
                           "run_p99_ms": distribution([r["p99_ms"] for r in runs]) if all("p99_ms" in r for r in runs) else None,
                           "observed_max_ms": max(r["max_ms"] for r in runs) if all("max_ms" in r for r in runs) else None,
                           "over_10ms": sum(r.get("over_10ms", 0) for r in runs),
                           "late": sum(r.get("late", 0) for r in runs),
                           "involuntary_switches": sum(r.get("involuntary_switches", 0) for r in runs),
                           "major_faults": sum(r.get("major_faults", 0) for r in runs)}
    return answer


def telemetry(cpu):
    result = {"loadavg": os.getloadavg(), "monotonic": time.monotonic()}
    for name in ("scaling_cur_freq", "scaling_governor", "energy_performance_preference"):
        p = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/{name}")
        if p.exists(): result[name] = p.read_text().strip()
    result["proc_stat"] = [s for s in Path("/proc/stat").read_text().splitlines()
                           if s.startswith("cpu")]
    result["thermal_throttle"] = {p.name: p.read_text().strip() for p in
        Path(f"/sys/devices/system/cpu/cpu{cpu}/thermal_throttle").glob("*") if p.is_file()}
    result["temperatures_millidegrees_c"] = {}
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        name = hwmon / "name"
        if name.exists() and name.read_text().strip() == "coretemp":
            result["temperatures_millidegrees_c"].update({str(p): p.read_text().strip() for p in hwmon.glob("temp*_input")})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--models", type=int, nargs="+", choices=(2, 8), default=[2, 8])
    p.add_argument("--blocks", type=int, default=24)
    p.add_argument("--hops", type=int, default=1000)
    p.add_argument("--mode", choices=("paced", "continuous"), default="paced")
    p.add_argument("--spectra", type=Path, nargs="+")
    p.add_argument("--cpu", type=int, required=True)
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--family-size", type=int, default=1,
                   help="Bonferroni-adjust CI alpha for this many planned comparisons")
    p.add_argument("--label", default="candidate")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.blocks < 4 or args.blocks % 2 or not 100 <= args.hops <= 1000000 or args.family_size < 1:
        p.error("Use an even block count >=4, 100..1000000 hops and family-size>=1")
    if args.cpu not in os.sched_getaffinity(0): p.error("CPU is outside allowed affinity")
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Prevent accidentally overlapping our own benchmark jobs, builds remain separate.
    lock = (ROOT / "validation" / "local" / "comparison.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if out.exists() or out.with_suffix(".jsonl").exists():
            raise FileExistsError("Refuse to overwrite previous evidence")
        builds = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
        identities = {v: {"directory": str(d), "benchmark_sha256": sha(d / "dpdfnet_benchmark"),
                          "library_sha256": sha(d / "libdpdfnet.so"),
                          "cmake_cache": (d / "CMakeCache.txt").read_text()}
                      for v, d in builds.items()}
        sources = [s.resolve() for s in args.spectra] if args.spectra else [None]
        manifest = {str(s): sha(s) for s in sources if s is not None}
        metadata = {"schema_version": 2, "label": args.label, "builds": identities,
                    "analysis_code_sha256": sha(Path(__file__)),
                    "platform": platform.platform(), "cpu": args.cpu,
                    "cpuinfo": Path("/proc/cpuinfo").read_text(), "input_sha256": manifest,
                    "protocol": vars(args) | {"warmup_hops": 120,
                       "all_runs_retained": True, "uncertainty_unit": "four-process block",
                       "primary_ci": "Paired Student t interval for mean log ratios",
                       "secondary_ci": "Percentile block bootstrap sensitivity analysis",
                       "randomization": "Balanced ABBA/BAAB assignments, exact or Monte Carlo",
                       "limitation": "Single host/session; thermal, scheduling and ASLR effects remain"}}
        rng = random.Random(args.seed)
        schedule = []
        for model in args.models:
            orders = ["ABBA", "BAAB"] * (args.blocks // 2)
            rng.shuffle(orders)
            for block, order in enumerate(orders):
                schedule.append((model, block, order, sources[block % len(sources)]))
        rng.shuffle(schedule)
        metadata["schedule"] = schedule
        out.with_suffix(".protocol.json").write_text(json.dumps(metadata, default=str, indent=2) + "\n")
        records = []
        with out.with_suffix(".jsonl").open("x") as journal:
            for model, block, order, source in schedule:
                for position, letter in enumerate(order):
                    variant = "baseline" if letter == "A" else "candidate"
                    weights = ROOT / "models" / f"dpdfnet{model}_48khz_hr" / "weights.f32"
                    command = ["taskset", "-c", str(args.cpu), str(builds[variant] / "dpdfnet_benchmark"),
                               str(model), str(weights), str(args.hops), args.mode]
                    if source: command.append(str(source))
                    before = telemetry(args.cpu)
                    env = os.environ.copy()
                    for name in ("LD_PRELOAD", "DPDF_SAMPLE", "DPDF_PERF_CONTROL", "DPDF_PERF_ACK"):
                        env.pop(name, None)
                    env["DPDF_BENCH_SAMPLES"] = str(out.parent / f"{out.stem}-m{model}-b{block}-{position}.f64")
                    result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
                    row = json.loads(result.stdout)
                    row.update(variant=variant, block=block, model_id=model, position=position,
                               order=order, source=str(source), before=before, after=telemetry(args.cpu))
                    records.append(row); journal.write(json.dumps(row) + "\n"); journal.flush()
                print(f"{args.label}: model {model}, block {block + 1}/{args.blocks} {order}", flush=True)
        for variant, directory in builds.items():
            if sha(directory / "libdpdfnet.so") != identities[variant]["library_sha256"]:
                raise RuntimeError("Library changed during collection")
        report = metadata | {"models": {str(m): summarize([r for r in records if r["model_id"] == m],
                                seed=args.seed, alpha=0.05 / args.family_size) for m in args.models}}
        out.write_text(json.dumps(report, default=str, indent=2) + "\n")
        for model, summary in report["models"].items():
            print(model, json.dumps(summary), flush=True)


if __name__ == "__main__": main()
