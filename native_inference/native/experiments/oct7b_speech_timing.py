"""Matched continuous inference timing on a saved real EARS-WHAM mixture.

One calling thread, preallocated C calls; causal FFT outside the timing region.
This complements the longer synthetic cadence confirmation, not quality scoring.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from statistics import median
import sys

from oct7b_optimization import ROOT, baseline_identity, prepare, sha, manifest
sys.path.insert(0,str(ROOT/'native'))
from extended_probe import ExtendedModel
from probe import audio_spectra, session
from optimization_probe import exact_recurrent_parity
from further_streaming_timing import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant')
    p.add_argument('--size',type=int,choices=(2,8),required=True)
    p.add_argument('--clip',default='00033')
    args=p.parse_args()
    source,library=baseline_identity(args.size)
    candidate=prepare(args.size,args.variant)
    model=ROOT/f'models/dpdfnet{args.size}_48khz_hr.onnx'
    weights=ROOT/f'models/rework{args.size}/weights.f32'
    clips=json.loads((ROOT/'scratch/fullband/evaluation/manifest.json').read_text())['clips']
    clip=next(item for item in clips if item['id']==args.clip)
    audio=ROOT/'scratch/fullband'/clip['noisy']
    frames=audio_spectra(audio)
    assert len(frames)>=1100, 'Real clip must contain 11 seconds including warmup'
    frames=frames[:1100]
    reference=session(model)
    builds={'baseline':library.parent,'candidate':ROOT/f'build/oct7b_{args.size}_{args.variant}'}
    models={name:ExtendedModel(reference,4,8,7,build=target,weights=weights) for name,target in builds.items()}
    report={'generated_at':datetime.now(timezone.utc).isoformat(),'model_size':args.size,
            'variant':args.variant,'source_manifest':manifest(candidate),
            'driver_sha256':sha(__file__),'timing_helper_sha256':sha(Path(__file__).with_name('further_streaming_timing.py')),
            'model_sha256':sha(model),'weights_sha256':sha(weights),
            'library_sha256':{name:sha(target/'libdpdf_full.so') for name,target in builds.items()},
            'clip':clip,'input_sha256':sha(audio),'sample_rate':48000,
            'method':'Four balanced continuous runs, 100 warmup and 1000 timed hops; preallocated single-thread C calls; real noisy EARS-WHAM PCM; FFT excluded.'}
    try:
        report['parity']=exact_recurrent_parity(models['baseline'],models['candidate'],frames[:1000])
        report['runs']=run(models,frames,4,False,100)
        report['summary']={}
        for name in models:
            rows=[item['implementations'][name] for item in report['runs']]
            item={key:median(row['wall'][key] for row in rows) for key in ('mean_ms','p50_ms','p99_ms')}
            item['max_ms']=max(row['wall']['max_ms'] for row in rows)
            item['over_10ms']=sum(row['wall']['over_10ms'] for row in rows)
            report['summary'][name]=item
        report['mean_reduction_percent']=100*(1-report['summary']['candidate']['mean_ms']/report['summary']['baseline']['mean_ms'])
    finally:
        for instance in models.values(): instance.close()
    assert report['library_sha256']=={name:sha(target/'libdpdf_full.so') for name,target in builds.items()}
    output=ROOT/f'results/oct7b_{args.size}_speech_timing.json'
    assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'model':args.size,'clip':args.clip,'summary':report['summary'],
                      'mean_reduction_percent':report['mean_reduction_percent']}),flush=True)


if __name__=='__main__': main()
