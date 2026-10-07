"""Compile and run the independent dense-padding oracle; save raw evidence."""
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
    for name in ('baseline','candidate','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cc',default='cc')
    p.add_argument('--sanitize',action='store_true',
                   help='Use with ASan/UBSan baseline and candidate libraries')
    args=p.parse_args()
    args.baseline=args.baseline.resolve(); args.candidate=args.candidate.resolve()
    args.output=args.output.resolve(); args.output.parent.mkdir(parents=True,exist_ok=True)
    compiler=shutil.which(args.cc)
    if not compiler:
        raise RuntimeError('Missing C compiler '+args.cc)
    source=Path(__file__).with_name('dense_padding_contract.c')
    log=args.output.with_suffix('.log')
    report={'generated_at':datetime.now(timezone.utc).isoformat(),
            'driver_sha256':sha(__file__),'harness_sha256':sha(source),
            'libraries':{'baseline':{'path':str(args.baseline),'sha256':sha(args.baseline)},
                         'candidate':{'path':str(args.candidate),'sha256':sha(args.candidate)}},
            'compiler':{'path':compiler,'sha256':sha(compiler),
                        'version':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]},
            'sanitizers':args.sanitize,
            'n_values':[1,7,8,16,24,31,32,64,192],
            'k_values':[1,7,8,17,64,127,512],
            'row_values':[1,2,4,5,47,48,49,96,97],
            'precision_modes':[0,16,8],
            'note':'Unsupported FP16/INT8 modes are skipped on scalar-only builds. '
                   'INT8 aliases cover <=48 rows or n<=k, preventing overwrite of future chunks. '
                   'Rounding/denormal controls change after identical nearest-mode creation.'}
    transcript=[]
    try:
        with tempfile.TemporaryDirectory(prefix='oct7b_dense_contract_') as folder:
            executable=Path(folder)/'dense_padding_contract'
            command=[compiler,'-O2','-std=c11','-Wall','-Wextra','-Werror',
                     '-ffp-contract=off',str(source),'-ldl','-lm','-o',str(executable)]
            if args.sanitize:
                command[1:1]=['-fsanitize=address,undefined','-fno-omit-frame-pointer','-no-pie']
            report['compile_command']=command
            compiled=subprocess.run(command,capture_output=True,text=True)
            transcript.append('Compile:\n'+compiled.stdout+compiled.stderr)
            report['compile_returncode']=compiled.returncode
            if compiled.returncode:
                raise RuntimeError('Dense oracle compilation failed')
            report['executable_sha256']=sha(executable)
            ran=subprocess.run([str(executable),str(args.baseline),str(args.candidate)],capture_output=True,text=True)
            transcript.append('Run:\n'+ran.stdout+ran.stderr)
            report['run_returncode']=ran.returncode
            if ran.returncode:
                raise RuntimeError('Dense oracle failed')
            report['checks']=json.loads(ran.stdout.splitlines()[-1])
            report['passed']=True
    except Exception as error:
        report['passed']=False; report['error']=str(error)
        raise
    finally:
        log.write_text('\n'.join(transcript))
        report['log_sha256']=sha(log)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'checks':report['checks']}),flush=True)


if __name__=='__main__':
    main()
