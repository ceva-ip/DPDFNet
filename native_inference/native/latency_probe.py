"""Balanced paired latency comparisons with wall and calling-thread CPU time.

Paired cadence runs both implementations for each input hop, reversing order
every hop. This controls slow host drift but has a higher duty cycle than a
single production stream. Standalone cadence should also be measured.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time
from extended_probe import ExtendedModel, CONFIGS
from probe import cpu_name, initial_state, session, spectra, summarize
from optimization_probe import exact_recurrent_parity


def paired(models, frames, repeats, paced, warmup):
    reports=[]
    for repeat in range(repeats):
        states={name: initial_state(m) for name,m in models.items()}
        values={name: {'wall': [], 'cpu': [], 'off_cpu': [], 'late_calls': [], 'switch_calls': 0} for name in models}
        lateness=[]
        start=time.perf_counter()
        for index,frame in enumerate(frames):
            deadline=start+index*.01
            if paced:
                delay=deadline-time.perf_counter()
                if delay>0: time.sleep(delay)
            if paced and index>=warmup:
                lateness.append(max(0,time.perf_counter()-deadline)*1000)
            names=list(models)
            if (index+repeat)%2: names.reverse()
            for name in names:
                before=resource.getrusage(resource.RUSAGE_THREAD)
                cpu_start=time.thread_time_ns()
                wall_start=time.perf_counter_ns()
                _,states[name]=models[name].run(None,{'spec':frame,'state_in':states[name]})
                wall=(time.perf_counter_ns()-wall_start)/1e6
                cpu=(time.thread_time_ns()-cpu_start)/1e6
                after=resource.getrusage(resource.RUSAGE_THREAD)
                if index<warmup: continue
                switches=(after.ru_nvcsw-before.ru_nvcsw)+(after.ru_nivcsw-before.ru_nivcsw)
                v=values[name]
                v['wall'].append(wall);v['cpu'].append(cpu);v['off_cpu'].append(max(0,wall-cpu))
                v['switch_calls']+=int(switches>0)
                if wall>10:
                    v['late_calls'].append({'frame':index,'wall_ms':wall,'thread_cpu_ms':cpu,'context_switches':switches})
        report={'repeat':repeat,'paced':paced,'implementations':{}}
        for name,v in values.items():
            report['implementations'][name]={
                'wall':summarize(v['wall']),'thread_cpu':summarize(v['cpu']),
                'off_cpu_ms_total':sum(v['off_cpu']),'calls_with_context_switches':v['switch_calls'],
                'over_10ms_details':v['late_calls']}
        if paced: report['paired_start_lateness']=summarize(lateness)
        reports.append(report)
        print(json.dumps(report),flush=True)
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--baseline-build',type=Path,required=True)
    p.add_argument('--candidate-build',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=1000)
    p.add_argument('--warmup',type=int,default=100)
    p.add_argument('--repeats',type=int,default=4)
    p.add_argument('--paced-repeats',type=int,default=4)
    p.add_argument('--standalone-paced-repeats',type=int,default=0)
    p.add_argument('--config',choices=CONFIGS,default='fc_and_1x1_8')
    p.add_argument('--cpu',type=int)
    a=p.parse_args()
    if a.frames<=0 or a.warmup<0 or min(a.repeats,a.paced_repeats,a.standalone_paced_repeats)<0:
        p.error('Invalid frame/repeat counts')
    if a.cpu is not None: os.sched_setaffinity(0,{a.cpu})
    ref=session(a.model)
    models={name:ExtendedModel(ref,*CONFIGS[a.config],build=build,weights=a.weights)
            for name,build in [('baseline',a.baseline_build),('candidate',a.candidate_build)]}
    try:
        parity=exact_recurrent_parity(models['baseline'],models['candidate'],spectra(500))
        frames=spectra(a.frames+a.warmup)
        report={'environment':{'cpu':cpu_name(),'platform':platform.platform(),'affinity':sorted(os.sched_getaffinity(0))},
                'method':'Per-hop AB/BA paired calls, CPU-time and unfiltered wall-time distributions; paired paced duty is two models per hop.',
                'model':str(a.model),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
                'weights_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest(),
                'builds':{'baseline':str(a.baseline_build),'candidate':str(a.candidate_build)},
                'config':a.config,'timed_frames_per_run':a.frames,'warmup':a.warmup,'parity':parity,
                'artifacts':{name:hashlib.sha256((build/'libdpdf_full.so').read_bytes()).hexdigest()
                             for name,build in [('baseline',a.baseline_build),('candidate',a.candidate_build)]},
                'owned_bytes':{name:m.owned_bytes for name,m in models.items()},
                'continuous':paired(models,frames,a.repeats,False,a.warmup),
                'paced':paired(models,frames,a.paced_repeats,True,a.warmup)}
        report['standalone_paced']=[]
        for repeat in range(a.standalone_paced_repeats):
            names=list(models)
            if repeat%2: names.reverse()
            for name in names:
                item=paired({name:models[name]},frames,1,True,a.warmup)[0]
                item['repeat']=repeat
                report['standalone_paced'].append(item)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2)+'\n')
    finally:
        for m in models.values(): m.close()


if __name__=='__main__': main()
