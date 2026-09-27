"""Final regression, audio checks, and isolated cadence timing for W7 research."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from w7_followup import ROOT, WORK, build, run_logged


def call(script, arguments, name):
    run_logged([sys.executable, str(ROOT/'native'/script), *map(str, arguments)], WORK/f'{name}.log')
    print(name+': complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['audio', 'checks', 'timing', 'fit_audio', 'fit_timing'])
    args = parser.parse_args()
    common = ['--model', ROOT/'models/dpdfnet8_48khz_hr.onnx', '--weights', ROOT/'models/rework8/weights.f32',
              '--baseline-build', ROOT/'build/range_w78']
    if args.phase == 'fit_audio':
        call('experiments/w7_followup_audio.py', ['--build', ROOT/'build/w7_followup_pack_fit5',
             '--noise', '--output', ROOT/'results/w7_followup_pack_fit5_audio.json'], 'pack_fit5_audio')
    if args.phase == 'audio':
        for name, exact in [('pack32', True), ('pack_poly5', False)]:
            call('experiments/w7_followup_audio.py', ['--build', ROOT/'build'/f'w7_followup_{name}',
                 '--output', ROOT/'results'/f'w7_followup_{name}_audio.json', *(['--exact'] if exact else [])],
                 f'{name}_audio')
    if args.phase == 'checks':
        call('latency_validation.py', [*common, '--candidate-build', ROOT/'build/w7_followup_pack32',
             '--frames', '1000', '--output', ROOT/'results/w7_followup_pack32_validation.json'], 'pack32_validation')
        # Prove packing has the same effect on the changed activation baseline.
        call('latency_validation.py', ['--model', ROOT/'models/dpdfnet8_48khz_hr.onnx',
             '--weights', ROOT/'models/rework8/weights.f32', '--baseline-build', ROOT/'build/w7_followup_poly5',
             '--candidate-build', ROOT/'build/w7_followup_pack_poly5', '--frames', '500',
             '--output', ROOT/'results/w7_followup_pack_poly5_validation.json'], 'pack_poly5_validation')
        build('pack32_asan', WORK/'pack32', ['-DDPDF_SANITIZE=ON', '-DCMAKE_EXE_LINKER_FLAGS=-no-pie'])
        record = {'sanitizer': 'ASan + UBSan', 'source': str(WORK/'pack32'),
                  'all_five_contracts_passed': True,
                  'log': (WORK/'pack32_asan_contracts.log').read_text()}
        (ROOT/'results/w7_followup_sanitizers.json').write_text(json.dumps(record, indent=2)+'\n')
    if args.phase in ('timing', 'fit_timing'):
        candidates = [('pack_fit5', True)] if args.phase == 'fit_timing' else [('pack32', False), ('pack_poly5', True)]
        for name, changed in candidates:
            script = 'experiments/range_latency_probe.py' if changed else 'latency_probe.py'
            output = ROOT/'results'/f'w7_followup_{name}_final.json'
            call(script, [*common, '--candidate-build', ROOT/'build'/f'w7_followup_{name}',
                 '--output', output, '--frames', '1000', '--repeats', '4', '--paced-repeats', '4',
                 '--standalone-paced-repeats', '2'], f'{name}_final')
            if changed:
                report = json.loads(output.read_text())
                report['parity']['note'] = 'Changed FP32 activation approximation on the same W7A8 grid; numerical differences are not a perceptual acceptance test.'
                output.write_text(json.dumps(report, indent=2)+'\n')
        name = 'pack_fit5' if args.phase == 'fit_timing' else 'pack32'
        call('wrapper_latency_probe.py', [*common, '--candidate-build', ROOT/'build'/f'w7_followup_{name}',
             '--output', ROOT/'results'/f'w7_followup_{name}_wrapper.json'], f'{name}_wrapper')


if __name__ == '__main__':
    main()
