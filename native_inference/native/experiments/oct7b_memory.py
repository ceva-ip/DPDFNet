"""Fresh-process warmed RSS, current Oct7 baseline versus a new candidate.

Run separately from builds, scoring and timing. Reuse the unchanged memory
worker and four balanced fresh children per implementation.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys

from oct7_memory import ROOT, PROBE, sha, source_identity, local_dependencies
from oct7b_optimization import baseline_identity, prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant')
    p.add_argument('--size',type=int,choices=(2,8),required=True)
    args = p.parse_args()
    source, library = baseline_identity(args.size)
    candidate = prepare(args.size,args.variant)
    builds = {'baseline':library.parent,
              'candidate':ROOT/f'build/oct7b_{args.size}_{args.variant}'}
    identities = {name:sha(build/'libdpdf_full.so') for name,build in builds.items()}
    model = ROOT/f'models/dpdfnet{args.size}_48khz_hr.onnx'
    weights = ROOT/f'models/rework{args.size}/weights.f32'
    output = ROOT/f'results/oct7b_{args.size}_{args.variant}_memory.json'
    assert not output.exists(), 'Use a fresh variant; saved memory results are immutable'
    report = {'generated_at':datetime.now(timezone.utc).isoformat(),
              'model_size':args.size,'variant':args.variant,
              'baseline':'Committed Oct7 combo_asm_norm',
              'protocol':'Four balanced sequential fresh processes per implementation; '
                         'source weights unmapped; 120 warmup hops; warmed RSS minus '
                         'common imported-runtime RSS; no reference ORT session created.',
              'library_sha256':identities,'model_sha256':sha(model),
              'weights_sha256':sha(weights),'driver_sha256':sha(__file__),
              'source_identity':{'baseline':source_identity(source),
                                 'candidate':source_identity(candidate)},
              'dependency_sha256':local_dependencies(PROBE),
              'raw_processes':[],'variants':{name:{'runs':[]} for name in builds}}
    env = dict(os.environ)
    env.update({name:'1' for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS',
                                    'MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    for repeat in range(4):
        for name in (list(builds) if repeat%2==0 else list(reversed(builds))):
            command = [sys.executable,str(PROBE),'--model',str(model),
                       '--weights',str(weights),'--build',str(builds[name]),
                       '--variant','selective_int8']
            process = subprocess.run(command,cwd=ROOT,env=env,capture_output=True,
                                     text=True,timeout=60)
            raw = {'repeat':repeat,'variant':name,'command':command,
                   'returncode':process.returncode,'stdout':process.stdout,
                   'stderr':process.stderr}
            report['raw_processes'].append(raw)
            if process.returncode:
                output.write_text(json.dumps(report,indent=2)+'\n')
                raise RuntimeError(process.stderr)
            measurement = json.loads(process.stdout)
            assert measurement['rss_after_bytes']-measurement['rss_before_bytes']==measurement['rss_delta_bytes']
            report['variants'][name]['runs'].append(measurement)
            print(json.dumps({'repeat':repeat,'variant':name,**measurement}),flush=True)
    assert identities=={name:sha(build/'libdpdf_full.so') for name,build in builds.items()}
    for item in report['variants'].values():
        item['median_incremental_rss_bytes'] = median(run['rss_delta_bytes'] for run in item['runs'])
        owned = {run['owned_bytes'] for run in item['runs']}
        assert len(owned)==1
        item['owned_bytes'] = owned.pop()
    report['completed'] = True
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in item.items() if k!='runs'}
                      for name,item in report['variants'].items()}),flush=True)


if __name__=='__main__':
    main()
