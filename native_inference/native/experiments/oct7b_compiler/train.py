"""Flush GCC PGO profiles from deterministic, synthetic-only spectral streams.

Invocation is a parent/worker pair. The worker exits before the parent hashes
profiles, since shared-library GCC counters are flushed on process termination.
No EARS, robustness or quality-scored file is loaded for training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'native'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transforms import profile_dir


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest(source):
    return {p.name: sha(p) for p in sorted(source.iterdir()) if p.is_file()
            and (p.suffix in ('.c','.h','.S') or p.name=='CMakeLists.txt')}


def worker(args):
    # Native BLAS environment is also forced by the parent before imports.
    import numpy as np
    from extended_probe import ExtendedModel
    from probe import session, initial_state, ptr, spectra
    reference = session(args.model)
    model = ExtendedModel(reference,4,8,7,build=args.build,weights=args.weights)
    base = spectra(args.frames)
    rng = np.random.default_rng(2026100702)
    # Valid real-FFT bins for an additional procedural noise stream. Its
    # distribution is only compiler training, never a perceptual evaluation.
    noise = rng.standard_normal(base.shape).astype(np.float32)*np.float32(.3)
    noise[:,0,0,0,1] = 0
    noise[:,0,0,-1,1] = 0
    streams = [('probe_default',base),
               ('probe_quiet',base*np.float32(.001)),
               ('probe_high',base*np.float32(8)),
               ('procedural_complex_noise',noise)]
    state = initial_state(reference)
    output = np.empty((1,1,481,2),np.float32)
    state_pointer, output_pointer = ptr(state),ptr(output)
    call = model.lib.dpdf_model_process
    records = []
    try:
        for label, frames in streams:
            state[...] = initial_state(reference)
            digest = hashlib.sha256()
            for frame in frames:
                digest.update(frame.tobytes())
                rc = call(model.handle,ptr(frame),state_pointer,output_pointer,state_pointer)
                if rc:
                    raise RuntimeError(f'{label}: process returned {rc}')
                if not np.isfinite(output).all() or not np.isfinite(state).all():
                    raise RuntimeError(f'{label}: nonfinite training output/state')
            records.append({'label':label,'frames':len(frames),
                            'spectral_input_sha256':digest.hexdigest(),
                            'final_output_sha256':hashlib.sha256(output.tobytes()).hexdigest(),
                            'final_state_sha256':hashlib.sha256(state.tobytes()).hexdigest()})
    finally:
        model.close()
    report = {'generated_at':datetime.now(timezone.utc).isoformat(),
              'method':'One calling thread, preallocated in-place state, four independent synthetic streams.',
              'scored_fixture_inputs_used':False,'config':{'tier':4,'precision':8,'mask':7},
              'worker_pid':os.getpid(),'model_sha256':sha(args.model),
              'weights_sha256':sha(args.weights),'instrumented_library_sha256':sha(args.build/'libdpdf_full.so'),
              'source_manifest':source_manifest(args.source),'streams':records}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Trained {len(streams)*args.frames} single-thread synthetic hops',flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model','weights','build','source','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frames',type=int,default=1000,
                   help='Hops per independent stream (default: 4,000 total)')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.frames<=0:
        p.error('Positive frames required')
    for name in ('model','weights','build','source','output'):
        setattr(args,name,getattr(args,name).resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.worker:
        worker(args)
        return
    profiles = profile_dir(args.build)
    if profiles.exists() and list(profiles.rglob('*.gcda')):
        raise RuntimeError('Training requires fresh empty counters; do not run instrumented CTest first')
    cache = (args.build/'CMakeCache.txt').read_text()
    expected_mode = 'DPDF_OCT7B_PGO_MODE:STRING=GENERATE'
    if expected_mode not in cache:
        raise RuntimeError('Expected an instrumented GENERATE build')
    expected_profile = 'DPDF_OCT7B_PROFILE_DIR:PATH='+str(profiles)
    if expected_profile not in cache:
        raise RuntimeError('Profile path does not match this target directory')
    env = os.environ.copy()
    env.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    command = [sys.executable,str(Path(__file__).resolve())]
    for name in ('model','weights','build','source','output'):
        command += ['--'+name,str(getattr(args,name))]
    command += ['--frames',str(args.frames),'--worker']
    subprocess.run(command,env=env,check=True)
    profile_files = sorted(profiles.rglob('*.gcda'))
    if not profile_files:
        raise RuntimeError('No GCC counters flushed after worker exit')
    report = json.loads(args.output.read_text())
    report.update({'driver_sha256':sha(__file__),
                   'transform_sha256':sha(Path(__file__).with_name('transforms.py')),
                   'environment':{'platform':platform.platform(),
                                  'affinity':sorted(os.sched_getaffinity(0))},
                   'profile_dir':str(profiles),
                   'profiles':{str(p.relative_to(profiles)):sha(p) for p in profile_files},
                   'counter_flush':'Worker process exited before profile collection'})
    compiler_line = next((v for v in cache.splitlines() if v.startswith('CMAKE_C_COMPILER:FILEPATH=')),None)
    if compiler_line:
        compiler = Path(compiler_line.partition('=')[2])
        report['compiler'] = {'path':str(compiler),'sha256':sha(compiler),
            'version':subprocess.check_output([str(compiler),'--version'],text=True).splitlines()[0]}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Collected {len(profile_files)} GCC counter files: {args.output}',flush=True)


if __name__=='__main__':
    main()
