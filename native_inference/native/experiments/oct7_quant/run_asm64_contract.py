"""Build/run the independent guard-page oracle against a frozen selected ASM.

Correctness only; no latency or RSS is collected. Linux x86-64, existing
development toolchain. Run only after model latency measurements finish.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=(2, 8), default=8)
    parser.add_argument('--cc', default='cc')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if sys.platform!='linux' or platform.machine() not in ('x86_64', 'AMD64'):
        parser.error('Direct SysV assembly guard-page oracle requires Linux x86-64')
    source = Path(__file__).with_name('asm64_contract.c')
    snapshot = ROOT/f'scratch/oct7/{args.size}/combo_asm_norm'
    assembly = snapshot/'qdot64_fixed_avx2.S'
    saved = json.loads((snapshot/'source_manifest.json').read_text())
    if sha(assembly) != saved['qdot64_fixed_avx2.S']:
        raise ValueError('Selected frozen assembler differs from its source manifest')
    frozen = {}
    for size in (2, 8):
        folder = ROOT/f'scratch/oct7/{size}/combo_asm_norm'
        assembly_path = folder/'qdot64_fixed_avx2.S'
        manifest_path = folder/'source_manifest.json'
        expected = json.loads(manifest_path.read_text())['qdot64_fixed_avx2.S']
        if sha(assembly_path) != expected or expected != sha(assembly):
            raise ValueError(f'Model {size} frozen assembler differs from the tested source')
        frozen[str(size)] = {'assembly_path': str(assembly_path), 'assembly_sha256': expected,
                             'source_manifest_sha256': sha(manifest_path)}
    executable = ROOT/f'build/oct7_{args.size}_asm64_contract'
    output = args.output or ROOT/'results/oct7_asm64_contract.json'
    logs = ROOT/f'scratch/oct7/{args.size}/asm64_guard'
    logs.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which(args.cc)
    if compiler is None:
        raise RuntimeError(f'Missing compiler: {args.cc}')
    compiler_version = subprocess.check_output([args.cc, '--version'], text=True)
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'method': 'Direct frozen assembly call versus independent scalar int32 dot; guarded mmap buffers and byte canaries.',
        'model_size_snapshot': args.size,
        'selected_variant': 'combo_asm_norm',
        'runtime_model_linked': False,
        'asan_instruments_assembly': False,
        'source_sha256': {'contract_c': sha(source), 'assembly': sha(assembly),
                          'runner': sha(__file__),
                          'snapshot_manifest': sha(snapshot/'source_manifest.json')},
        'source_paths': {'contract_c': str(source), 'assembly': str(assembly)},
        'both_model_frozen_assembly': frozen,
        'selected_library_sha256': {name: sha(ROOT/f'build/oct7_{args.size}_combo_asm_norm'/name)
                                    for name in ('libdpdf_full.so', 'libdpdf_dprnn.so')},
        'environment': {'platform': platform.platform(), 'compiler': compiler,
                        'compiler_sha256': sha(compiler), 'compiler_version': compiler_version},
        'stages': [], 'passed': False,
    }
    build = [args.cc, '-O2', '-std=c11', '-Wall', '-Wextra', '-Werror',
             '-fno-tree-vectorize', '-ffp-contract=off', str(source), str(assembly),
             '-o', str(executable)]
    try:
        for name, command in (('build', build), ('contract', [str(executable)])):
            done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
            log = logs/(name+'.log')
            log.write_text(done.stdout+done.stderr)
            stage = {'name': name, 'command': command, 'returncode': done.returncode,
                     'stdout': done.stdout, 'stderr': done.stderr,
                     'log_path': str(log), 'log_sha256': sha(log)}
            report['stages'].append(stage)
            if done.returncode:
                raise RuntimeError(f'{name} failed with return code {done.returncode}; {log}')
            if name=='build':
                report['executable_sha256'] = sha(executable)
            else:
                report['contract'] = json.loads(done.stdout)
                if report['contract'].get('passed') is not True:
                    raise ValueError('Direct assembly contract did not pass')
        if (sha(source)!=report['source_sha256']['contract_c'] or
                sha(assembly)!=report['source_sha256']['assembly'] or
                sha(__file__)!=report['source_sha256']['runner']):
            raise RuntimeError('Contract/assembler/runner source changed during checking')
        report['passed'] = True
    except Exception as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        output.write_text(json.dumps(report, indent=2)+'\n')
        raise
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'passed': True, **report['contract']}), flush=True)


if __name__=='__main__':
    main()
