#!/usr/bin/env python3
"""Compare every frame/state using the independent local fixture manifest."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time
import os

ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(job):
    runner,baseline,candidates,model,weights,case=job
    result=json.loads(subprocess.check_output([runner,baseline,candidates[0],str(model),weights,case['path'],*candidates[1:]],text=True))
    if result['hops']!=case['hops'] or result['model']!=model or result['candidates']!=len(candidates):
        raise ValueError('Worker coverage mismatch')
    return result|{'id':case['id']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,nargs='+',required=True)
    p.add_argument('--runner',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--models',type=int,nargs='+',default=[2,8])
    p.add_argument('--case-limit',type=int)
    p.add_argument('--workers',type=int,default=1,help='Parallel correctness jobs; never overlap with timing runs')
    p.add_argument('--cpus',type=int,nargs='+',help='Allowed CPUs for correctness workers')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); start=time.monotonic()
    if args.workers<1:p.error('--workers must be positive')
    if args.output.exists() or args.output.with_suffix('.jsonl').exists():raise FileExistsError(args.output)
    if args.cpus:
        if not set(args.cpus)<=os.sched_getaffinity(0):p.error('CPUs outside allowed affinity')
        os.sched_setaffinity(0,set(args.cpus))
    manifest=json.loads(args.manifest.read_text());cases=manifest['cases']
    if args.case_limit: cases=cases[:args.case_limit]
    records=[];jobs=[]
    for case in cases:
        if sha(Path(case['path']))!=case['sha256']: raise ValueError('Input checksum mismatch')
        for model in args.models:
            weights=ROOT/'models'/f'dpdfnet{model}_48khz_hr'/'weights.f32'
            jobs.append((str(args.runner.resolve()),str(args.baseline.resolve()),[str(x.resolve()) for x in args.candidate],model,str(weights),case))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool,args.output.with_suffix('.jsonl').open('x') as journal:
        futures=[pool.submit(verify,job) for job in jobs]
        try:
            for future in as_completed(futures):
                result=future.result();records.append(result)
                journal.write(json.dumps(result)+'\n');journal.flush()
                print(result['model'],result['id'],result['hops'],flush=True)
        except BaseException:
            for future in futures:future.cancel()
            raise
    records.sort(key=lambda x:(x['model'],x['id']))
    report={'schema_version':2,'baseline_sha256':sha(args.baseline),'candidates':[{'path':str(x.resolve()),'sha256':sha(x)} for x in args.candidate],
            'manifest_sha256':sha(args.manifest),'historical_suite':False,'results':records,
            'workers':args.workers,'correctness_not_performance':True,
            'elapsed_seconds':time.monotonic()-start,'all_passed':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
