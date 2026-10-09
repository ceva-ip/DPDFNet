#!/usr/bin/env python3
"""Dependency-free build driver for the supported GCC/Linux release."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(*arguments):
    subprocess.run([str(x) for x in arguments], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path, default=ROOT / "build")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--no-pgo", action="store_true")
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    build = args.build_dir.resolve()
    for size in (2, 8):
        folder = ROOT / "models" / ("dpdfnet%d_48khz_hr" % size)
        metadata = json.loads((folder / "manifest.json").read_text())
        digest = hashlib.sha256((folder / "weights.f32").read_bytes()).hexdigest()
        if digest != metadata["weights_sha256"]:
            raise RuntimeError("Weight checksum mismatch: %s" % folder)
    settings = ["-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
                "-DDPDF_SANITIZE=" + ("ON" if args.sanitize else "OFF")]
    if not args.no_pgo and not args.sanitize:
        # Every training run has a fresh directory; never combine stale profiles.
        profiles = build / "profiles" / uuid.uuid4().hex
        run("cmake", "-S", ROOT, "-B", build, *settings,
            "-DBUILD_TESTING=OFF", "-DDPDF_PGO_MODE=GENERATE", "-DDPDF_PROFILE_DIR=" + str(profiles))
        run("cmake", "--build", build, "--parallel", args.jobs, "--target", "dpdfnet_train")
        run(build / "dpdfnet_train", ROOT / "models/dpdfnet8_48khz_hr/weights.f32")
        if not list(profiles.rglob("*.gcda")):
            raise RuntimeError("Training did not produce profiles")
        run("cmake", "-S", ROOT, "-B", build, *settings,
            "-DBUILD_TESTING=ON", "-DDPDF_PGO_MODE=USE", "-DDPDF_PROFILE_DIR=" + str(profiles))
    else:
        run("cmake", "-S", ROOT, "-B", build, *settings,
            "-DBUILD_TESTING=ON", "-DDPDF_PGO_MODE=OFF")
    run("cmake", "--build", build, "--parallel", args.jobs)
    run("ctest", "--test-dir", build, "--output-on-failure", "--parallel", "1")


if __name__ == "__main__":
    main()
