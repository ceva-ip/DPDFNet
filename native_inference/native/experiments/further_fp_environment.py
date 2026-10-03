"""Exact single-thread W7A8 parity after changing caller FP controls post-create.

Linux x86_64 validation only. Models are created before testing all four
rounding modes with FTZ/DAZ off/on. Every case advances independent recurrent
streams for 32 frames by default and uses spectrum/state in place. This checks
outputs and return status; floating-point exception flags are not compared.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FP = ct.POINTER(ct.c_float)


HELPER = r"""
#include <fenv.h>
#include <xmmintrin.h>
static fenv_t saved;
static int valid;
int fp_save(void) { valid=(fegetenv(&saved)==0); return valid ? 0 : -1; }
int fp_restore(void) { return valid ? fesetenv(&saved) : -1; }
int fp_configure(int index,int denorm) {
    const int modes[4]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
    if (index<0 || index>3 || (denorm!=0 && denorm!=1)) return -1;
    if (fesetround(modes[index])) return -1;
    unsigned mxcsr=(_mm_getcsr() & ~0x8040u) | (denorm ? 0x8040u : 0u);
    _mm_setcsr(mxcsr);
    return feclearexcept(FE_ALL_EXCEPT);
}
unsigned fp_mxcsr(void) { return _mm_getcsr(); }
int fp_round(void) { return fegetround(); }
"""


def pointer(array):
    return array.ctypes.data_as(FP)


class Model:
    def __init__(self, build, weights):
        self.path = build / "libdpdf_full.so"
        self.lib = ct.CDLL(str(self.path.resolve()))
        self.lib.dpdf_model_weights_sha256.restype = ct.c_char_p
        self.lib.dpdf_model_weight_count.restype = ct.c_size_t
        self.lib.dpdf_model_state_size.restype = ct.c_size_t
        self.lib.dpdf_model_create_config.argtypes = [FP, ct.c_size_t, ct.c_int, ct.c_int, ct.c_uint]
        self.lib.dpdf_model_create_config.restype = ct.c_void_p
        self.lib.dpdf_model_init_state.argtypes = [FP]
        self.lib.dpdf_model_init_state.restype = ct.c_int
        self.lib.dpdf_model_process.argtypes = [ct.c_void_p, FP, FP, FP, FP]
        self.lib.dpdf_model_process.restype = ct.c_int
        self.lib.dpdf_model_destroy.argtypes = [ct.c_void_p]
        self.lib.dpdf_model_destroy.restype = None
        data = weights.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.lib.dpdf_model_weights_sha256().decode():
            raise ValueError("Weight SHA mismatch")
        array = np.frombuffer(data, dtype="<f4")
        if array.size != self.lib.dpdf_model_weight_count():
            raise ValueError("Weight size mismatch")
        self.handle = self.lib.dpdf_model_create_config(pointer(array), array.size, 4, 8, 7)
        if not self.handle:
            raise RuntimeError(f"Could not create {self.path}")
        self.state = np.empty(self.lib.dpdf_model_state_size(), dtype=np.float32)
        self.frame = np.empty((1, 1, 481, 2), dtype=np.float32)

    def reset(self):
        if self.lib.dpdf_model_init_state(pointer(self.state)):
            raise RuntimeError("State reset failed")

    def process(self, frame):
        np.copyto(self.frame, frame)
        rc = self.lib.dpdf_model_process(self.handle, pointer(self.frame), pointer(self.state),
                                         pointer(self.frame), pointer(self.state))
        if rc:
            raise RuntimeError(f"Process failed for {self.path}: {rc}")
        return self.frame.tobytes(), self.state.tobytes()

    def close(self):
        if self.handle:
            self.lib.dpdf_model_destroy(self.handle)
            self.handle = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-build", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/rework8/weights.f32")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cc", default="cc")
    parser.add_argument("--verify-single-thread", action="store_true",
                        help="Require one OS thread; launch with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1")
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("Positive frame count required")
    scratch = ROOT / "scratch/further_validation"
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="fp_environment_", dir=str(scratch)))
    source = work / "fp_environment.c"
    source.write_bytes(HELPER.encode())
    shared = work / "fp_environment.so"
    subprocess.run([args.cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC",
                    "-shared", str(source), "-o", str(shared), "-lm"], check=True)
    controls = ct.CDLL(str(shared))
    for name in ("fp_save", "fp_restore", "fp_round"):
        getattr(controls, name).argtypes = []
        getattr(controls, name).restype = ct.c_int
    controls.fp_configure.argtypes = [ct.c_int, ct.c_int]
    controls.fp_configure.restype = ct.c_int
    controls.fp_mxcsr.argtypes = []
    controls.fp_mxcsr.restype = ct.c_uint
    if controls.fp_save():
        raise RuntimeError("Could not save caller FP environment")
    models = {}
    thread_counts = {len(list(Path('/proc/self/task').iterdir()))}
    report = {
        "frames_per_mode": args.frames,
        "modes": [],
        "post_creation_control_changes": True,
        "in_place_spectrum_and_state": True,
        "helper_source": str(source),
        "helper_sha256": hashlib.sha256(HELPER.encode()).hexdigest(),
        "exception_flags": "Floating-point exception flags are not compared.",
        "sources": [
            "https://codebrowser.dev/glibc/glibc/sysdeps/x86_64/fpu/fegetenv.c.html",
            "https://codebrowser.dev/glibc/glibc/sysdeps/x86_64/fpu/fesetenv.c.html",
        ],
    }
    try:
        if controls.fp_configure(0, 0):
            raise RuntimeError("Could not select default FP controls")
        # Generate inputs once under default controls, before test environments.
        rng = np.random.default_rng(71951)
        frames = rng.normal(0, .03, (args.frames, 1, 1, 481, 2)).astype(np.float32)
        frames[::4] *= np.float32(1e-5)
        frames[:, 0, 0, 0, 0] = np.float32(1e-40)
        frames[:, 0, 0, 0, 1] = np.float32(-1e-40)
        builds = {"baseline": args.baseline_build}
        builds.update({f"candidate{index}": build for index, build in enumerate(args.candidate_build)})
        for name, build in builds.items():
            models[name] = Model(build, args.weights)
            thread_counts.add(len(list(Path('/proc/self/task').iterdir())))
        report["builds"] = {name: {"path": str(model.path),
                                  "sha256": hashlib.sha256(model.path.read_bytes()).hexdigest()}
                            for name, model in models.items()}
        report["model_creation_mxcsr"] = hex(controls.fp_mxcsr())
        names = ["nearest", "downward", "upward", "towardzero"]
        for rounding, name in enumerate(names):
            for denorm in (0, 1):
                for model in models.values():
                    model.reset()
                mode = {"rounding": name, "ftz_daz": bool(denorm), "exact": True,
                        "frames_checked": 0, "finite": True}
                for index, frame in enumerate(frames):
                    reference = None
                    for label, model in models.items():
                        # Reapply before each implementation, so it gets the same
                        # controls even after another call raises exception flags.
                        if controls.fp_configure(rounding, denorm):
                            raise RuntimeError("Could not select FP controls")
                        result = model.process(frame)
                        thread_counts.add(len(list(Path('/proc/self/task').iterdir())))
                        if reference is None:
                            reference = result
                        elif result != reference:
                            mode.update(exact=False, failure_candidate=label,
                                        first_mismatch_frame=index,
                                        output_equal=result[0] == reference[0],
                                        state_equal=result[1] == reference[1])
                            break
                        if not np.isfinite(model.frame).all() or not np.isfinite(model.state).all():
                            mode["finite"] = False
                            break
                    if not mode["exact"] or not mode["finite"]:
                        break
                    mode["frames_checked"] += 1
                mode["caller_mxcsr"] = hex(controls.fp_mxcsr())
                report["modes"].append(mode)
                print(json.dumps(mode), flush=True)
        report["passed"] = all(mode["exact"] and mode["finite"] and
                               mode["frames_checked"] == args.frames for mode in report["modes"])
    finally:
        # Restore even on exceptions, including failures in model construction.
        restored = controls.fp_restore() == 0
        for model in models.values():
            model.close()
        thread_counts.add(len(list(Path('/proc/self/task').iterdir())))
        if not restored:
            raise RuntimeError("Could not restore original caller FP environment")
    report['observed_process_thread_counts'] = sorted(thread_counts)
    report['single_thread_verification_requested'] = args.verify_single_thread
    if args.verify_single_thread:
        report['passed'] = report['passed'] and thread_counts == {1}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("FP environment parity or single-thread verification failed; see JSON report")


if __name__ == "__main__":
    main()
