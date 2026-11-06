#!/usr/bin/env python3
"""Profile only warmed inference using perf's FIFO control protocol.

Use --sudo if permissions require it. Only perf keeps root privileges; its
workload drops to the invoking user's UID/GID and clears capabilities.
No sysctl, file capabilities, or system installation is changed by this tool.
Timing claims must come from separate runs without perf attached.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from compare_benchmarks import sha

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-dir',type=Path,required=True)
    p.add_argument('--perf',type=Path,default=Path('/usr/bin/perf'))
    p.add_argument('--perf-library-path',type=Path)
    p.add_argument('--sudo',action='store_true')
    p.add_argument('--model',type=int,choices=(2,8),required=True)
    p.add_argument('--cpu',type=int,required=True)
    p.add_argument('--hops',type=int,default=5000)
    p.add_argument('--spectra',type=Path,required=True)
    p.add_argument('--kind',choices=('stat','record'),default='stat')
    p.add_argument('--events',help='Defaults to cycles:u for record, four generic counters for stat')
    p.add_argument('--frequency',type=int,default=997)
    p.add_argument('--output',type=Path,required=True,help='Prefix for raw data, log, worker JSON and metadata')
    args=p.parse_args()
    events=args.events or ('cycles:u' if args.kind=='record' else '{cycles:u,instructions:u,branches:u,branch-misses:u}')
    if args.hops<100 or args.cpu not in os.sched_getaffinity(0): p.error('Invalid hops or CPU')
    out=args.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
    control=out.with_suffix('.ctl');ack=out.with_suffix('.ack')
    worker=out.with_suffix('.worker.json');log=out.with_suffix('.log')
    data=out.with_suffix('.data' if args.kind=='record' else '.stdout')
    for f in (control,ack,worker,log,data,out.with_suffix('.meta.json')):
        if f.exists(): raise FileExistsError(f)
    worker.touch()  # Child redirection retains the ordinary user's ownership.
    os.mkfifo(control);os.mkfifo(ack)
    variables=['DPDF_PERF_CONTROL='+str(control),'DPDF_PERF_ACK='+str(ack)]
    if args.perf_library_path: variables.append('LD_LIBRARY_PATH='+str(args.perf_library_path.resolve()))
    command=(['sudo','-n'] if args.sudo else [])+['env',*variables,str(args.perf.resolve()),args.kind,
             '--delay=-1','--control=fifo:'+str(control)+','+str(ack)]
    if args.kind=='stat': command+=['-j','-e',events]
    else: command+=['--no-buildid-cache','-e',events,'-F',str(args.frequency),'-o','-']
    workload=[]
    if args.sudo:
        workload+=['setpriv','--reuid='+str(os.getuid()),'--regid='+str(os.getgid()),'--init-groups',
                   '--bounding-set=-all','--inh-caps=-all','--ambient-caps=-all']
    executable=args.build_dir.resolve()/'dpdfnet_benchmark'
    library=args.build_dir.resolve()/'libdpdfnet.so'
    weights=ROOT/'models'/f'dpdfnet{args.model}_48khz_hr'/'weights.f32'
    workload+=['taskset','-c',str(args.cpu),str(executable),str(args.model),str(weights),str(args.hops),
               'continuous',str(args.spectra.resolve())]
    # Redirection happens before dropping privileges, into a precreated user file.
    command+=['--','sh','-c',shlex.join(workload)+' > '+shlex.quote(str(worker))]
    start=time.monotonic()
    try:
        with data.open('wb') as stdout,log.open('wb') as stderr:
            result=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=max(180,args.hops*.02))
        report={'command':command,'returncode':result.returncode,'elapsed_seconds':time.monotonic()-start,
                'worker_uid':os.getuid(),'worker_gid':os.getgid(),'hops':args.hops,'warmup_hops':120,
                'counter_window':'FIFO enable after warmup; disable after timed hops',
                'library_sha256':sha(library),'benchmark_sha256':sha(executable),
                'spectra_sha256':sha(args.spectra),'weights_sha256':sha(weights),
                'perf_sha256':sha(args.perf),'raw_data_sha256':sha(data),'log_sha256':sha(log)}
        out.with_suffix('.meta.json').write_text(json.dumps(report,indent=2)+'\n')
        if result.returncode: raise RuntimeError(log.read_text(errors='replace'))
        print(json.dumps(report,indent=2))
    finally:
        control.unlink(missing_ok=True);ack.unlink(missing_ok=True)

if __name__=='__main__':main()
