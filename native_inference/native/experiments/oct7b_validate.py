"""Validation and safety evidence for a selected Oct7b candidate.

No timing or memory collection here. Run after building release, ASan and
scalar targets and separately from benchmark jobs.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys

from oct7b_optimization import ROOT, baseline_identity, prepare, sha, manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('variant')
    p.add_argument('--size',type=int,choices=(2,8),required=True)
    p.add_argument('--stage',choices=('audio','compatibility','fp','contracts','dense'),required=True)
    args=p.parse_args()
    source, library=baseline_identity(args.size)
    candidate=prepare(args.size,args.variant)
    build=ROOT/f'build/oct7b_{args.size}_{args.variant}'
    model=ROOT/f'models/dpdfnet{args.size}_48khz_hr.onnx'
    weights=ROOT/f'models/rework{args.size}/weights.f32'
    prefix=ROOT/f'results/oct7b_{args.size}'
    def run(script,arguments):
        subprocess.run([sys.executable,str(ROOT/'native'/script),*map(str,arguments)],check=True,cwd=ROOT)
    if args.stage=='audio':
        output=Path(str(prefix)+f'_{args.variant}_audio.json')
        assert not output.exists()
        arguments=['--model',model,'--weights',weights,'--baseline-build',library.parent,
                   '--candidate-build',build,'--baseline-source',source,'--candidate-source',candidate,
                   '--workers','4','--output',output]
        if args.size==2:
            arguments+=['--model-quality-report',ROOT/'results/dpdfnet2_overview_quality.json']
        run('experiments/further_exact_validation.py',arguments)
        report=json.loads(output.read_text())
        old=json.loads((ROOT/f'results/oct7_{args.size}_combo_asm_norm_audio.json').read_text())
        previous={item['case']:item for item in old['cases']}
        assert len(report['cases'])==65
        for item in report['cases']:
            assert item['pcm_sha256']['candidate']==previous[item['case']]['pcm_sha256']['candidate']
            assert item['stream_sha256']['candidate']==previous[item['case']]['stream_sha256']['candidate']
        report['all_65_saved_oct7_baseline_stream_and_pcm_hashes_match']=True
        output.write_text(json.dumps(report,indent=2)+'\n')
    elif args.stage=='compatibility':
        run('latency_validation.py',['--model',model,'--weights',weights,'--baseline-build',library.parent,
                                    '--candidate-build',build,'--configs','compact_fp32','fc_and_1x1_16',
                                    'fc_and_1x1_8','all_8','--frames','1000','--output',Path(str(prefix)+'_compatibility.json')])
    elif args.stage=='fp':
        run('experiments/further_fp_environment.py',['--baseline-build',library.parent,'--candidate-build',build,
                    '--weights',weights,'--frames','64','--verify-single-thread',
                    '--output',Path(str(prefix)+'_fp_environment.json')])
    elif args.stage=='dense':
        for suffix in ('','_asan'):
            arguments=['--baseline',library.parent.with_name(library.parent.name+suffix)/'libdpdf_full.so',
                       '--candidate',build.with_name(build.name+suffix)/'libdpdf_full.so',
                       '--output',Path(str(prefix)+f'_selected_dense{suffix}.json')]
            if suffix:
                arguments+=['--sanitize']
            run('experiments/oct7b_dense/run_contract.py',arguments)
    else:
        report={'model_size':args.size,'variant':args.variant,
                'generated_at':datetime.now(timezone.utc).isoformat(),'driver_sha256':sha(__file__),
                'source_manifest':manifest(candidate),'stages':[]}
        for suffix in ('','_asan','_scalar'):
            target=build.with_name(build.name+suffix)
            command=['ctest','--test-dir',str(target),'--output-on-failure']
            done=subprocess.run(command,capture_output=True,text=True,cwd=ROOT)
            tests=json.loads(subprocess.check_output(['ctest','--test-dir',str(target),'--show-only=json-v1'],text=True))
            report['stages'].append({'build':str(target),'suffix':suffix,'command':command,
                'returncode':done.returncode,'stdout':done.stdout,'stderr':done.stderr,
                'registered_tests':[item['name'] for item in tests['tests']],
                'library_sha256':sha(target/'libdpdf_full.so'),
                'direct_assembly_oracles':{},'assembly_is_not_asan_instrumented':True})
            if done.returncode:
                Path(str(prefix)+'_contracts.json').write_text(json.dumps(report,indent=2)+'\n')
                raise RuntimeError(done.stdout+done.stderr)
            for executable in ('dpdf_fused_contract','dpdf_pair_contract','dpdf_wide_contract'):
                if (target/executable).is_file():
                    checked=subprocess.run([str(target/executable)],capture_output=True,text=True,check=True)
                    report['stages'][-1]['direct_assembly_oracles'][executable]={
                        'executable_sha256':sha(target/executable),'stdout':checked.stdout,
                        'stderr':checked.stderr,'checks':json.loads(checked.stdout)}
            print(args.size,args.variant,suffix,report['stages'][-1]['registered_tests'],flush=True)
        report['passed']=True
        Path(str(prefix)+'_contracts.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
