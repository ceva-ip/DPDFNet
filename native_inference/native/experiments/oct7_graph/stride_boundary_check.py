"""Run the original layout oracle with additional stride-boundary widths.

Creates a temporary source/executable and leaves baseline sources untouched.
No timing samples or latency statistics are collected. Invoke after benchmark
jobs finish, because correctness calls still consume CPU.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[3]
OLD_WIDTHS = "const int widths[]={1,7,8,9,16,17,40,48,65,80,96,160,480};"
NEW_WIDTHS = "const int widths[]={18,19,34,35};"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(args):
    build = args.build.resolve()
    library = build / "libdpdf_full.so"
    if not library.is_file():
        raise FileNotFoundError(library)
    original_path = ROOT / "native/layout_contract.c"
    original = original_path.read_bytes()
    code = original.decode()
    if code.count(OLD_WIDTHS) != 1:
        raise ValueError("The original layout oracle width declaration changed")
    code = code.replace(OLD_WIDTHS, NEW_WIDTHS, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".log")
    report = {
        "passed": False,
        "method": "Original scalar per-output convolution oracle; only width fixture declaration changed.",
        "widths": [18, 19, 34, 35],
        "convolution_shapes": 108,
        "scalar_and_avx2": True,
        "other_original_transpose_and_norm_contracts_retained": True,
        "original_oracle": str(original_path),
        "original_source_sha256": sha(original),
        "generated_source_sha256": sha(code.encode()),
        "library": str(library),
        "library_sha256": sha(library.read_bytes()),
        "driver_sha256": sha(Path(__file__).read_bytes()),
        "log": str(log_path),
        "sanitizers": bool(args.sanitize),
    }
    compiler = subprocess.run([args.cc, "--version"], capture_output=True, text=True)
    report["compiler"] = compiler.stdout.splitlines()[0] if compiler.stdout else args.cc
    transcript = []
    try:
        with tempfile.TemporaryDirectory(prefix="oct7_stride_boundary_") as temporary:
            folder = Path(temporary)
            source = folder / "layout_contract.c"
            target = folder / "layout_contract"
            source.write_bytes(code.encode())
            command = [args.cc, "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
                       "-ffp-contract=off", "-DDPDF_X86_DISPATCH",
                       "-I", str(ROOT / "native"), str(source), str(library),
                       f"-Wl,-rpath,{build}", "-lm", "-o", str(target)]
            if args.sanitize:
                command.extend(["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-no-pie"])
            report["compile_command"] = command
            compiled = subprocess.run(command, capture_output=True, text=True)
            transcript.append("Compile:\n" + compiled.stdout + compiled.stderr)
            report["compile_returncode"] = compiled.returncode
            if compiled.returncode:
                raise RuntimeError("Boundary oracle compilation failed")
            executed = subprocess.run([str(target)], capture_output=True, text=True)
            transcript.append("Execute:\n" + executed.stdout + executed.stderr)
            report["run_returncode"] = executed.returncode
            if executed.returncode:
                raise RuntimeError("Boundary oracle execution failed")
            report["passed"] = True
            print(executed.stdout, end="", flush=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        log_path.write_text("\n".join(transcript))
        report["log_sha256"] = sha(log_path.read_bytes())
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--sanitize", action="store_true",
                        help="Link/run the oracle against an ASan/UBSan candidate build")
    run(parser.parse_args())
