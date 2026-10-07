"""Record direct fixed-pair assembler/scalar-oracle/guard-page evidence."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Frozen candidate source snapshot')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path,help='Optional selected binary identity for provenance')
    p.add_argument('--cc',default='cc')
    p.add_argument('--sanitize',action='store_true')
    args=p.parse_args(); args.source=args.source.resolve(); args.output=args.output.resolve()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    compiler=shutil.which(args.cc)
    if not compiler: raise RuntimeError('Missing compiler '+args.cc)
    source=args.source/'pair_contract.c'; assembly=args.source/'qdot4pair_fixed_avx2.S'
    saved=json.loads((args.source/'source_manifest.json').read_text())
    for path in (source,assembly):
        if sha(path)!=saved[path.name]: raise RuntimeError('Frozen source differs: '+path.name)
    include96='dpdf_qdot4pair96_avx2:' in assembly.read_text()
    report={'generated_at':datetime.now(timezone.utc).isoformat(),'passed':False,
            'method':'Direct assembler versus independent scalar int32 dot; read-only inputs, guard pages and canaries.',
            'runtime_model_linked':False,'asan_instruments_assembly':False,
            'widths':[64,96] if include96 else [64],
            'source_sha256':{'contract':sha(source),'assembly':sha(assembly),'runner':sha(__file__)},
            'compiler':{'path':compiler,'sha256':sha(compiler),
                        'version':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]},
            'sanitizers':args.sanitize}
    if args.library: report['library_sha256']=sha(args.library)
    log=args.output.with_suffix('.log'); transcript=[]
    try:
        with tempfile.TemporaryDirectory(prefix='oct7b_pair_contract_') as folder:
            executable=Path(folder)/'pair_contract'
            command=[compiler,'-O2','-std=c11','-Wall','-Wextra','-Werror','-fno-tree-vectorize',
                     '-ffp-contract=off',str(source),str(assembly),'-o',str(executable)]
            if include96: command.insert(1,'-DDPDF_PAIR_TEST96')
            if args.sanitize: command[1:1]=['-fsanitize=address,undefined','-fno-omit-frame-pointer','-no-pie']
            report['compile_command']=command
            built=subprocess.run(command,capture_output=True,text=True)
            transcript.append('Compile:\n'+built.stdout+built.stderr); report['compile_returncode']=built.returncode
            if built.returncode: raise RuntimeError('Pair oracle compilation failed')
            report['executable_sha256']=sha(executable)
            ran=subprocess.run([str(executable)],capture_output=True,text=True)
            transcript.append('Run:\n'+ran.stdout+ran.stderr); report['run_returncode']=ran.returncode
            if ran.returncode: raise RuntimeError('Pair oracle failed')
            report['checks']=json.loads(ran.stdout.splitlines()[-1]); report['passed']=True
    except Exception as error:
        report['error']=str(error); raise
    finally:
        log.write_text('\n'.join(transcript)); report['log_sha256']=sha(log)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'checks':report['checks']}),flush=True)


if __name__=='__main__': main()
