#!/usr/bin/env python3
"""Fixed-size studies using the prepared lab; standard-library dependencies only."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics as stats
import subprocess
import sys
import time

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from compare_benchmarks import student_t_quantile
CONTROLLER = Path(__file__).resolve().with_name('controller.py')
ROOT = TOOLS.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def control(*args):
    try:
        output=subprocess.check_output(['sudo','-n','/usr/bin/python3',str(CONTROLLER),*map(str,args)], text=True)
    except subprocess.CalledProcessError as error:
        # The controller emits its structured failure and rollback verdict to
        # stdout. Preserve it in the session log instead of hiding it inside
        # CalledProcessError.output when preparation fails before ownership.
        detail=(error.output or '').strip()
        raise RuntimeError('Laboratory controller failed: '+(detail or str(error))) from error
    return json.loads(output)


def describe(values):
    a=sorted(values);n=len(a)
    return dict(n=n,mean=stats.mean(a),sd=stats.stdev(a) if n>1 else None,
                min=a[0],p95=a[min(n-1,int(n*.95))],p99=a[min(n-1,int(n*.99))],max=a[-1])


def effect(logs,alpha):
    mean=stats.mean(logs)
    half=student_t_quantile(1-alpha/2,len(logs)-1)*stats.stdev(logs)/math.sqrt(len(logs))
    return dict(gain_percent=100*(1-math.exp(mean)),interval_percent=[100*(1-math.exp(mean+half)),100*(1-math.exp(mean-half))],
                paired_log_sd=stats.stdev(logs),paired_blocks=len(logs))


def condition_specs(m):
    return m.get('conditions', [dict(id='batch',worker_mode='batch',batch_size=m.get('batch_size',16),calls=m.get('calls',{}).get('batch',512)),
                                dict(id='paced',worker_mode='paced',period_ns=10000000,batch_size=1,calls=m.get('calls',{}).get('paced',100))])


def condition(m,j):
    if 'worker_mode' in j:return j
    found=next((c for c in condition_specs(m) if c['id']==j['mode']),None)
    if found:return found
    return dict(worker_mode=j['mode'],batch_size=m['batch_size'] if j['mode']=='batch' else 1,calls=m['calls'][j['mode']])


def config(path):
    m=json.loads(Path(path).read_text())
    if m.get('schema_version')!=1:raise ValueError('Unsupported manifest')
    for key in ['worker','inputs','variants','models','weights','seed','calls','batch_size','aa_blocks','compare_blocks']:
        if key not in m:raise ValueError('Missing manifest field '+key)
    if not 4<=m['aa_blocks']<=16 or not 4<=m['compare_blocks']<=32:raise ValueError('Use bounded fixed block counts')
    if m['aa_blocks']%4 or m['compare_blocks']%4:raise ValueError('Blocks must balance both orders within two inputs (multiple of four)')
    if len(m['inputs'])!=2 or len(set(m['inputs']))!=2:raise ValueError('This balanced plan requires two distinct input files')
    if not m['models'] or len(set(m['models']))!=len(m['models']) or not set(m['models'])<={2,8}:raise ValueError('Models must be unique IDs 2 and/or 8')
    if not m['variants']:raise ValueError('At least one ISA pair is required')
    if not isinstance(m['batch_size'],int) or not 1<=m['batch_size']<=256:raise ValueError('Invalid batch size')
    for mode in ['batch','continuous','paced']:
        n=m['calls'][mode]
        if not isinstance(n,int) or not 16<=n<=1000000:raise ValueError('Invalid call count')
    if m['calls']['batch']%m['batch_size'] or m['calls']['batch']//m['batch_size']<2:raise ValueError('Need at least two complete batches')
    for isa,arms in m['variants'].items():
        if set(arms)!={'baseline','candidate'}:raise ValueError('Need baseline and candidate for '+isa)
    specs=condition_specs(m)
    if len(set(c['id'] for c in specs))!=len(specs) or not specs:raise ValueError('Condition IDs must be unique')
    for c in specs:
        if c['worker_mode'] not in ['batch','continuous','paced']:raise ValueError('Unsupported condition mode')
        n,b=c['calls'],c.get('batch_size',1)
        if not isinstance(n,int) or not isinstance(b,int) or not 16<=n<=1000000 or not 1<=b<=256 or n%b or n//b<2:raise ValueError('Invalid condition size')
        if c['worker_mode']!='batch' and b!=1:raise ValueError('Individual-call mode requires batch size 1')
        if c['worker_mode']=='paced' and not 1000000<=c.get('period_ns',10000000)<=100000000:raise ValueError('Invalid cadence')
    family=len(m['models'])*len(specs)*len(m['variants'])
    if m.get('family_comparisons',family)<family:raise ValueError('Multiplicity family cannot be smaller than the planned contrasts')
    if m.get('power_profile','steady') not in ['steady','boost']:raise ValueError('Unknown power profile')
    binary_manifest(m)  # Verify all files before any privileged preparation.
    return m


def binary_manifest(m):
    paths=[m['worker'],*m['inputs'],*m['weights'].values()]
    paths += [p for arms in m['variants'].values() for p in arms.values()]
    paths += m.get('extra_files',[])
    for c in condition_specs(m):
        if c.get('worker'):paths.append(c['worker'])
        paths += c.get('inputs',[])
    return {str(Path(p).resolve()):sha(p) for p in paths}


def schedule(m,kind):
    rng=random.Random(m['seed']+(0 if kind=='aa' else 1000))
    blocks=m['aa_blocks'] if kind=='aa' else m['compare_blocks']
    isas=list(m['variants'])
    jobs=[]
    for block in range(blocks):
        cells=[(isa,model,mode) for isa in isas for model in m['models'] for mode in [c['id'] for c in condition_specs(m)]]
        rng.shuffle(cells)
        for isa,model,mode in cells:
            order=['A','B'] if (block//2+sum(map(ord,mode)))%2==0 else ['B','A']
            for label in order:
                spec=next(c for c in condition_specs(m) if c['id']==mode)
                inputs=spec.get('inputs',m['inputs'])
                arm='baseline' if kind=='aa' or label=='A' else 'candidate'
                jobs.append(dict(index=len(jobs),block=block,isa=isa,model=model,mode=mode,label=label,arm=arm,input=inputs[block%2]))
    return jobs


def worker_args(m,j,raw):
    c=condition(m,j);mode=c['worker_mode']
    if mode=='paced':mode='paced:'+str(c.get('period_ns',10000000))
    return [str(Path(c.get('worker',m['worker'])).resolve()),str(Path(m['variants'][j['isa']][j['arm']]).resolve()),str(j['model']),
            str(Path(m['weights'][str(j['model'])]).resolve()),str(Path(j['input']).resolve()),str(c['calls']),
            mode,str(c.get('batch_size',1)),str(raw.resolve())]


def measurement(m,j,out):
    raw=out/'samples'/f"{j['index']:04d}.f64"
    start=time.time_ns()
    result=control('exec','--',*worker_args(m,j,raw))
    end=time.time_ns()
    data=array.array('d');data.frombytes(raw.read_bytes())
    if len(data)!=result['observations'] or abs(stats.mean(data)-result['mean_ms'])>1e-9:raise RuntimeError('Raw observation mismatch')
    if abs(stats.stdev(data)-result['sample_sd_ms'])>1e-9:raise RuntimeError('SD mismatch')
    if result['uid']==0 or result['no_new_privs']!=1:raise RuntimeError('Privileged worker')
    if m.get('layout_profile','randomized')=='fixed' and not result.get('layout_randomization_disabled'):raise RuntimeError('ASLR policy not applied')
    if not result['hardware_feedback_available']:raise RuntimeError('Hardware frequency feedback unavailable')
    return dict(j,result=result,samples=str(raw.relative_to(out)),samples_sha256=sha(raw),started_ns=start,finished_ns=end)


def summarize(m,rows,kind,out):
    conditions=[];family=m.get('family_comparisons',len(m['models'])*len(condition_specs(m))*len(m['variants']))
    keys=sorted(set((r['isa'],r['model'],r['mode']) for r in rows))
    for isa,model,mode in keys:
        cell=[r for r in rows if (r['isa'],r['model'],r['mode'])==(isa,model,mode)]
        logs=[]
        for block in sorted(set(r['block'] for r in cell)):
            p={r['label']:r for r in cell if r['block']==block}
            logs.append(math.log(p['B']['result']['mean_ms']/p['A']['result']['mean_ms']))
        summary=effect(logs,.05/family)
        arms={}
        for label in ['A','B']:
            runs=[r for r in cell if r['label']==label];raw=[]
            for r in runs:
                a=array.array('d');a.frombytes((out/r['samples']).read_bytes());raw.extend(a)
            arms[label]=dict(run_means_ms=describe([r['result']['mean_ms'] for r in runs]),observations_ms=describe(raw),
                            observation_unit=runs[0]['result']['observation_unit'],
                            hardware_frequency_ratios=describe([r['result']['delivered_delta']/r['result']['reference_delta'] for r in runs]),
                            late=sum(r['result']['late'] for r in runs),over_10ms=sum(r['result']['over_10ms'] for r in runs),
                            minor_faults=sum(r['result']['minor_faults'] for r in runs),major_faults=sum(r['result']['major_faults'] for r in runs))
        summary.update(isa=isa,model=model,mode=mode,arms=arms)
        if kind=='aa':
            lo,hi=summary['interval_percent']
            summary['noise_envelope_percent']=max(abs(lo),abs(hi))
            summary['sensitivity_qualified']=summary['noise_envelope_percent']<=m.get('aa_noise_limit_percent',.5)
        conditions.append(summary)
    return dict(kind=kind,conditions=conditions,processes=len(rows),calls=sum(r['result']['calls'] for r in rows),
                no_observations_removed=True,unit_of_inference='paired fresh-process block',family_comparisons=family,
                alpha_family=.05,individual_interval_percent=100*(1-.05/family))


def timing(m,kind,path,aa=None):
    if kind=='compare' and (aa is None or aa.get('kind')!='aa' or aa.get('hashes')!=binary_manifest(m) or aa.get('manifest')!=m):
        raise ValueError('Run A/A with exactly these binaries and inputs before comparison')
    out=Path(path).resolve();out.mkdir(parents=True,exist_ok=False);(out/'samples').mkdir()
    before=control('ready');emit(out/'ready-before.json',before)
    jobs=schedule(m,kind)
    frozen=binary_manifest(m)
    emit(out/'plan.json',dict(kind=kind,manifest=m,hashes=frozen,schedule=jobs,created_ns=time.time_ns()))
    emit(out/'noise-before.json',control('status'))
    rows=[];checks={}
    with (out/'runs.jsonl').open('x') as journal:
        for j in jobs:
            row=measurement(m,j,out);rows.append(row)
            key=(j['model'],j['input'],j['mode'])
            checksum=row['result']['checksum']
            if key in checks and checksum!=checks[key]:raise RuntimeError('Candidate state/output checksum mismatch')
            checks[key]=checksum
            journal.write(json.dumps(row)+'\n');journal.flush()
            if len(rows)%8==0:print(kind,len(rows),'/',len(jobs),flush=True)
    if binary_manifest(m)!=frozen:raise RuntimeError('Binary or input changed during the study')
    emit(out/'noise-after.json',control('status'))
    emit(out/'ready-after.json',control('ready'))
    result=summarize(m,rows,kind,out)
    result['hashes']=frozen
    result['manifest']=m
    if kind=='compare':
        controls={(c['isa'],c['model'],c['mode']):c for c in aa['conditions']}
        for c in result['conditions']:
            a=controls[c['isa'],c['model'],c['mode']]
            c['aa_noise_envelope_percent']=a['noise_envelope_percent']
            c['aa_qualified']=a['sensitivity_qualified']
            c['noninferiority_margin_percent']=m.get('noninferiority_margin_percent',.5)
            c['passes_noninferiority']=c['aa_qualified'] and c['interval_percent'][0]>-c['noninferiority_margin_percent']
            c['passes_acceptance']=c['aa_qualified'] and c['interval_percent'][0]>max(a['noise_envelope_percent'],m.get('minimum_useful_gain_percent',.3)) and all(not arm['late'] and not arm['over_10ms'] and not arm['major_faults'] for arm in c['arms'].values())
    emit(out/'result.json',result)
    return result


def recover_aa(m,out):
    out=Path(out);plan=json.loads((out/'plan.json').read_text())
    if plan['manifest']!=m or plan['hashes']!=binary_manifest(m):raise ValueError('A/A recovery configuration changed')
    jobs=schedule(m,'aa');rows=[json.loads(line) for line in (out/'runs.jsonl').read_text().splitlines()]
    if len(rows)!=len(jobs):raise ValueError('Cannot resume an incomplete A/A collection')
    for row,job in zip(rows,jobs):
        if any(row[k]!=v for k,v in job.items()):raise ValueError('A/A schedule differs')
        raw=out/row['samples'];a=array.array('d');a.frombytes(raw.read_bytes())
        if sha(raw)!=row['samples_sha256'] or len(a)!=row['result']['observations']:raise ValueError('A/A raw data differ')
        if abs(stats.mean(a)-row['result']['mean_ms'])>1e-9 or abs(stats.stdev(a)-row['result']['sample_sd_ms'])>1e-9:raise ValueError('A/A metrics differ')
    result=summarize(m,rows,'aa',out);result.update(hashes=plan['hashes'],manifest=m,recovered_from_complete_raw_journal=True)
    emit(out/'result.json',result)
    return result


def diagnostics(m,out):
    out=Path(out);out.mkdir();(out/'samples').mkdir();rows=[]
    for isa in m['variants']:
        for model in m['models']:
            for arm in ['baseline','candidate']:
                j=dict(index=len(rows),isa=isa,model=model,arm=arm,mode='continuous',input=m['inputs'][0])
                rows.append(measurement(m,j,out))
    emit(out/'runs.json',rows)
    return rows


def profiles(m,path):
    out=Path(path).resolve();out.mkdir();(out/'samples').mkdir()
    groups={'primary':'{cycles:u,instructions:u,branches:u,branch-misses:u,ref-cycles:u}',
            'frontend':'{cycles:u,instructions:u,L1-icache-load-misses:u,L1-dcache-load-misses:u,dTLB-load-misses:u}'}
    jobs=[];rng=random.Random(m['seed']+2000)
    groups={k:v for k,v in groups.items() if k in m.get('profile_groups',list(groups))}
    selected=m.get('profile_conditions')
    specs=[c for c in condition_specs(m) if c['id'] in selected] if selected else [dict(id='batch',worker_mode='batch',batch_size=16,calls=2048)]
    for group in [*groups,*(['record'] if m.get('profile_sample',True) else [])]:
        for rep in range(1 if group=='record' else m.get('profile_repetitions',2)):
            cells=[(isa,model,arm,c) for isa in m['variants'] for model in m['models'] for arm in ['baseline','candidate'] for c in specs]
            rng.shuffle(cells)
            for isa,model,arm,c in cells:
                jobs.append(dict(index=len(jobs),isa=isa,model=model,arm=arm,mode=c['id'],input=c.get('inputs',m['inputs'])[0],group=group,repeat=rep,worker=c.get('worker',m['worker']),
                                 worker_mode=c['worker_mode'],batch_size=c.get('batch_size',1),period_ns=c.get('period_ns',10000000),
                                 calls=m.get('profile_calls',{}).get(c['id'],c['calls'])))
    emit(out/'plan.json',jobs);records=[]
    for j in jobs:
        stem=out/f"{j['index']:03d}";ctl=stem.with_suffix('.ctl');ack=stem.with_suffix('.ack')
        for p in [ctl,ack]:os.mkfifo(p,0o600)
        raw=out/'samples'/f"{j['index']:03d}.f64"
        env=['sudo','-n','env','LC_ALL=C','DPDF_PERF_CONTROL='+str(ctl),'DPDF_PERF_ACK='+str(ack)]
        cmd=[*env,'perf','record' if j['group']=='record' else 'stat','--delay=-1','--control=fifo:'+str(ctl)+','+str(ack)]
        if j['group']=='record':cmd+=['--no-buildid-cache','-e','cycles:u','-F','1999','-o',str(stem.with_suffix('.data'))]
        else:cmd+=['-j','-e',groups[j['group']]]
        profile_m=m
        cmd+=['--','/usr/bin/python3',str(CONTROLLER),'exec','--',*worker_args(profile_m,j,raw)]
        try:
            with stem.with_suffix('.log').open('w') as err,stem.with_suffix('.worker.json').open('w') as stdout:
                subprocess.run(cmd,stdout=stdout,stderr=err,check=True,timeout=60)
            row=dict(j,timing_excluded_from_speed_inference=True)
            if j['group']!='record':
                counters=[json.loads(line) for line in stem.with_suffix('.log').read_text().splitlines() if line.startswith('{')]
                if len(counters)!=5 or any(float(c['pcnt-running'])<99.9 for c in counters):raise RuntimeError('Unsupported/multiplexed PMU group')
                for c in counters:float(c['counter-value'])
                row['counters']=counters;row['calls']=j['calls']
            else:
                data=stem.with_suffix('.data');subprocess.run(['sudo','-n','chown',f'{os.getuid()}:{os.getgid()}',str(data)],check=True)
                with stem.with_suffix('.samples.txt').open('w') as f:subprocess.run(['perf','script','--show-lost-events','-i',str(data),'-F','comm,pid,tid,time,event,ip,sym,dso'],stdout=f,check=True)
                if 'PERF_RECORD_LOST' in stem.with_suffix('.samples.txt').read_text():raise RuntimeError('Lost hardware samples')
                row['data_sha256']=sha(data)
            records.append(row)
            with (out/'hardware.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('profile',len(records),'/',len(jobs),flush=True)
        finally:
            ctl.unlink(missing_ok=True);ack.unlink(missing_ok=True)
    emit(out/'result.json',dict(jobs=len(records),all_counters_running_100_percent=True,timing_excluded=True))
    return records


def report(out,aa,comparison):
    lines=['# Isolated quantize4 comparison','','All raw observations are retained. Batch samples are amortized group means, not individual-call latency. Process/block pairs are statistical replicates. Power/idle policy is recorded in prepare.json; fixed and dynamic frequency profiles must be interpreted separately.','',
           '| ISA | Model | Mode | Gain | Adjusted interval | Baseline mean ± run SD, µs | Candidate mean ± run SD, µs |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for c in comparison['conditions']:
        a=c['arms']['A']['run_means_ms'];b=c['arms']['B']['run_means_ms'];lo,hi=c['interval_percent']
        lines.append(f"| {c['isa']} | {c['model']} | {c['mode']} | {c['gain_percent']:+.3f}% | [{lo:+.3f},{hi:+.3f}]% | {a['mean']*1000:.3f} ± {a['sd']*1000:.3f} | {b['mean']*1000:.3f} ± {b['sd']*1000:.3f} |")
    lines+=['','## A/A sensitivity','','| Model | Mode | A/A gain | Adjusted interval | Noise envelope | Qualified at planned limit |','| --- | --- | ---: | ---: | ---: | --- |']
    for c in aa['conditions']:
        lo,hi=c['interval_percent'];lines.append(f"| {c['isa']} / {c['model']} | {c['mode']} | {c['gain_percent']:+.3f}% | [{lo:+.3f},{hi:+.3f}]% | {c['noise_envelope_percent']:.3f}% | {c['sensitivity_qualified']} |")
    lines+=['','No optional sample extension or discarded slow call. A comparison whose matching A/A condition is unqualified remains diagnostic. No default promotion follows from this report automatically. See JSON for sample SD, tails, frequency ratios, faults, deadline misses and binary/input hashes.','']
    (out/'RESULTS.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['aa','compare','profile','session','resume'])
    p.add_argument('manifest');p.add_argument('output');p.add_argument('--aa');a=p.parse_args();m=config(a.manifest)
    if a.mode in ['aa','compare']:
        timing(m,a.mode,a.output,json.loads(Path(a.aa).read_text()) if a.aa else None);return
    if a.mode=='profile':profiles(m,a.output);return
    out=Path(a.output).resolve()
    if a.mode=='resume':
        if not out.is_dir() or (out/'compare').exists():raise ValueError('Resume supports completed A/A with comparison not yet started')
        aa=recover_aa(m,out/'aa')
        archive=out/'diagnostics'/('aa-recovery-'+str(time.time_ns()));archive.mkdir(parents=True)
        for name in ['prepare.json','restore.json']:
            if (out/name).exists():(out/name).replace(archive/name)
    else:out.mkdir(parents=True,exist_ok=False)
    owned=False;session=None
    try:
        ready=control('prepare','--cpu',str(m.get('cpu',3)),'--frequency-khz',str(m.get('frequency_khz','auto')),
                      '--ttl',str(m.get('ttl_seconds',1800)),'--power-profile',m.get('power_profile','steady'),'--layout',m.get('layout_profile','randomized'),'--probe',str(Path(m['worker']).resolve()))
        owned=True;session=ready['session_id'];emit(out/'prepare.json',ready)
        if a.mode!='resume':aa=timing(m,'aa',out/'aa')
        comparison=timing(m,'compare',out/'compare',aa)
        if m.get('continuous_diagnostics',True):diagnostics(m,out/'continuous-diagnostics')
        if m.get('hardware_profiles',True):profiles(m,out/'profiles')
        emit(out/'ready-final.json',control('ready'))
        report(out,aa,comparison)
    finally:
        if owned:emit(out/'restore.json',control('restore','--session',session))


if __name__=='__main__':
    def stop(*args):raise KeyboardInterrupt('interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    main()
