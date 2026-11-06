#!/usr/bin/env python3
"""Balanced fresh-process hot-kernel throughput comparisons.

These are hot-cache amortized call times, not whole-model or per-call latency.
The worker warms each kernel for at least 100 ms before measurement.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
from compare_benchmarks import ROOT, sha, summarize, telemetry

ITERATIONS={'depthwise2':50000,'depthwise3':40000,'pair40':20000,'pair48':20000,
            'batch40':30000,'row64':1000000,'gates1':1000000,'gates40':30000,
            'gates48':30000,'norm40':50000,'transpose':1000000}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--models',type=int,nargs='+',choices=(2,8),default=[2,8])
    p.add_argument('--kernels',nargs='+',choices=tuple(ITERATIONS),required=True)
    p.add_argument('--blocks',type=int,default=12)
    p.add_argument('--cpu',type=int,required=True)
    p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--family-size',type=int,default=1)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.blocks<4 or args.blocks%2 or args.family_size<1: p.error('Use even blocks>=4 and family-size>=1')
    if args.cpu not in os.sched_getaffinity(0): p.error('Invalid CPU affinity')
    builds={'baseline':args.baseline.resolve(),'candidate':args.candidate.resolve()}
    identities={v:{str(m):sha(d/f'kernel_benchmark{m}') for m in args.models} for v,d in builds.items()}
    out=args.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() or out.with_suffix('.jsonl').exists(): raise FileExistsError(out)
    lock=ROOT/'validation/local/comparison.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    rng=random.Random(args.seed);schedule=[]
    for model in args.models:
        for kernel in args.kernels:
            orders=['ABBA','BAAB']*(args.blocks//2);rng.shuffle(orders)
            schedule.extend((model,kernel,block,order) for block,order in enumerate(orders))
    rng.shuffle(schedule)
    protocol={'arguments':vars(args),'schedule':schedule,'worker_sha256':identities,
              'metric':'hot-cache amortized call time; no per-call p99 or deadlines measured',
              'warmup':'At least 100 ms of the same kernel before timed calls',
              'analysis_sha256':sha(Path(__file__).with_name('compare_benchmarks.py'))}
    out.with_suffix('.protocol.json').write_text(json.dumps(protocol,default=str,indent=2)+'\n')
    records=[]
    with lock.open('w') as guard,out.with_suffix('.jsonl').open('x') as journal:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for model,kernel,block,order in schedule:
            checksum=None
            for position,letter in enumerate(order):
                variant='baseline' if letter=='A' else 'candidate'
                command=['taskset','-c',str(args.cpu),str(builds[variant]/f'kernel_benchmark{model}'),kernel,str(ITERATIONS[kernel])]
                before=telemetry(args.cpu)
                env=os.environ.copy();env.pop('LD_PRELOAD',None)
                row=json.loads(subprocess.check_output(command,text=True,env=env))
                if checksum is None: checksum=row['checksum']
                elif row['checksum']!=checksum: raise RuntimeError('Kernel result checksum mismatch')
                row.update(variant=variant,model_id=model,block=block,order=order,position=position,
                           before=before,after=telemetry(args.cpu))
                records.append(row);journal.write(json.dumps(row)+'\n');journal.flush()
            print(model,kernel,block+1,args.blocks,flush=True)
    summaries={}
    for model in args.models:
        for kernel in args.kernels:
            summary=summarize([r for r in records if r['model_id']==model and r['kernel']==kernel],
                              seed=args.seed,alpha=.05/args.family_size)
            summary['family_adjusted_randomization_p']=min(1,summary['two_sided_sign_randomization_p']*args.family_size)
            for variant in ('baseline','candidate'):
                for field in ('over_10ms','late'): summary[variant][field]=None
            summaries[f'{model}:{kernel}']=summary
    out.write_text(json.dumps(protocol|{'summaries':summaries},default=str,indent=2)+'\n')

if __name__=='__main__':main()
