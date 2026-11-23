# Reproducible local CPU laboratory. All mutations are reversible and journaled.
set shell := ["bash", "-euo", "pipefail", "-c"]
set positional-arguments

# Compile the common measurement/probe executable before isolating CPUs.
bench-tools:
    mkdir -p .local/bench-lab
    /usr/bin/gcc -O3 -Wall -Wextra -Werror -ffp-contract=off -mavx2 -mfma -Ioptimized_inference/include optimized_inference/tools/bench_lab/worker.c -ldl -lm -o .local/bench-lab/worker.tmp
    mv .local/bench-lab/worker.tmp .local/bench-lab/worker

# Reserve the discovered L3 domain and verify actual non-boost frequency.
bench-prepare cpu="3" frequency_khz="auto" ttl_seconds="1800" power_profile="steady": bench-tools
    sudo -n python3 optimized_inference/tools/bench_lab/controller.py prepare --cpu "$1" --frequency-khz "$2" --ttl "$3" --power-profile "$4" --probe "$(pwd)/.local/bench-lab/worker" | tee .local/bench-lab/prepare.json

# Refresh readiness, including three hardware-counter frequency probes.
bench-ready:
    sudo -n python3 optimized_inference/tools/bench_lab/controller.py ready | tee .local/bench-lab/ready.json

# Inspect current preparation/restoration state without another workload.
bench-status:
    sudo -n python3 optimized_inference/tools/bench_lab/controller.py status

# Run an absolute executable as the ordinary user on the reserved worker CPU.
bench-exec +command:
    sudo -n python3 optimized_inference/tools/bench_lab/controller.py exec -- "$@"

# Restore and independently read back every saved setting. Safe to repeat.
bench-restore:
    sudo -n python3 optimized_inference/tools/bench_lab/controller.py restore | tee .local/bench-lab/restore.json

# Behavioral tests do not modify the machine or require root.
bench-test:
    python3 optimized_inference/tests/bench_lab_test.py

# Fixed-size A/A control; refuses to reuse an existing output directory.
bench-aa manifest output:
    python3 optimized_inference/tools/bench_lab/study.py aa "$1" "$2"

# Comparison requires the unchanged-binary A/A report.
bench-compare manifest output aa_report:
    python3 optimized_inference/tools/bench_lab/study.py compare "$1" "$2" --aa "$3"

# Separate hardware counters and sampling; never used as latency results.
bench-profile manifest output:
    python3 optimized_inference/tools/bench_lab/study.py profile "$1" "$2"

# Recommended complete workflow: prepare, A/A, compare, profiles, finally restore.
bench-session manifest output: bench-tools
    python3 optimized_inference/tools/bench_lab/study.py session "$1" "$2"

# Continue after a fully collected A/A phase; never repeat collected samples.
bench-resume manifest output: bench-tools
    python3 optimized_inference/tools/bench_lab/study.py resume "$1" "$2"
