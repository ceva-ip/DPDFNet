#!/usr/bin/env python3
"""Repeat the C benchmark in fresh processes and summarize run-to-run latency."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def summarize(runs):
    """SD describes run means, not individual hops or uncertainty of the mean."""
    times = [r["mean_ms"] for r in runs]
    return {
        "runs": len(runs),
        "mean_ms": statistics.mean(times),
        "run_mean_sd_ms": statistics.stdev(times),
        "run_mean_min_ms": min(times),
        "run_mean_max_ms": max(times),
        "p99_ms_median": statistics.median(r["p99_ms"] for r in runs),
        "max_ms": max(r["max_ms"] for r in runs),
        "over_10ms": sum(r["over_10ms"] for r in runs),
        "late": sum(r["late"] for r in runs),
        "owned_bytes": runs[0]["owned_bytes"],
        "rss_increment_median_bytes": statistics.median(r["rss_increment_bytes"] for r in runs),
        "rss_total_median_bytes": statistics.median(r["rss_bytes"] for r in runs),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir", type=Path, default=ROOT / "build")
    p.add_argument("--model", choices=("2", "8", "all"), default="all")
    p.add_argument("--runs", type=int, default=32)
    p.add_argument("--hops", type=int, default=1000)
    p.add_argument("--mode", choices=("paced", "continuous"), default="paced")
    p.add_argument("--spectra", type=Path, nargs="+", help="Rotate precomputed real-audio spectrum files")
    p.add_argument("--output", type=Path, default=ROOT / "validation/local/benchmark.json")
    args = p.parse_args()
    if args.runs < 2 or not 100 <= args.hops <= 1000000:
        p.error("At least two runs and 100–1,000,000 hops are required")
    executable = args.build_dir.resolve() / "dpdfnet_benchmark"
    sizes = (2, 8) if args.model == "all" else (int(args.model),)
    records = {str(size): [] for size in sizes}
    for repeat in range(args.runs):
        for size in (sizes if repeat % 2 == 0 else sizes[::-1]):
            weights = ROOT / "models" / ("dpdfnet%d_48khz_hr" % size) / "weights.f32"
            command = [str(executable), str(size), str(weights), str(args.hops), args.mode]
            source = args.spectra[repeat % len(args.spectra)].resolve() if args.spectra else None
            if source: command.append(str(source))
            run = json.loads(subprocess.check_output(command, text=True))
            run.update(repeat=repeat, spectrum_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if source else None)
            records[str(size)].append(run)
            print("Model %d, run %d/%d: %.3f ms" % (size, repeat + 1, args.runs, run["mean_ms"]), flush=True)
    report = {
        "schema_version": 1,
        "benchmark_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256((args.build_dir.resolve() / "libdpdfnet.so").read_bytes()).hexdigest()
        if (args.build_dir.resolve() / "libdpdfnet.so").is_file() else None,
        "platform": platform.platform(),
        "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name")), "unknown"),
        "weights_sha256": {str(size): hashlib.sha256((ROOT / "models" /
                              ("dpdfnet%d_48khz_hr" % size) / "weights.f32").read_bytes()).hexdigest()
                           for size in sizes},
        "mode": args.mode, "warmup_hops": 120, "timed_hops_per_run": args.hops,
        "statistic": "Mean and sample SD (n-1) of fresh-process run means; not per-hop SD or a confidence interval",
        "models": {size: {"summary": summarize(runs), "runs": runs} for size, runs in records.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for size, value in report["models"].items():
        s = value["summary"]
        print("Model %s: %.3f ± %.3f ms (%d runs)" % (size, s["mean_ms"], s["run_mean_sd_ms"], s["runs"]))


if __name__ == "__main__": main()
