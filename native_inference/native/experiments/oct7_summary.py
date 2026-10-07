"""Collect the selected Oct7 evidence and verify links to the scored reference.

Run only after timing, correctness and fresh-process memory measurements finish.
This checks retained reports and hashes; it performs no inference or benchmark.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELECTED = 'combo_asm_norm'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(name):
    path = ROOT/'results'/name
    return json.loads(path.read_text()), {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}


def collect(size):
    final, final_id = record(f'oct7_{size}_{SELECTED}_final.json')
    audio, audio_id = record(f'oct7_{size}_{SELECTED}_audio.json')
    memory, memory_id = record(f'oct7_{size}_memory.json')
    prior_name = 'further2_best_single_audio.json' if size == 2 else 'further_best_single_audio.json'
    prior, prior_id = record(prior_name)
    assert audio['all_selected_cases_bit_identical'] and audio['full_available_suite']
    assert audio['selected_cases'] == 65
    assert memory['completed']
    assert audio['library_sha256'] == final['artifacts']
    assert audio['model_sha256'] == final['model_sha256']
    assert audio['weights_sha256'] == final['weights_sha256']
    assert prior['library_sha256']['candidate'] == final['artifacts']['baseline']
    assert prior['model_sha256'] == final['model_sha256']
    assert prior['weights_sha256'] == final['weights_sha256']
    assert final['parity']['output_and_state_bit_identical']
    assert final['owned_bytes']['candidate'] == final['owned_bytes']['baseline']
    for name in ('baseline', 'candidate'):
        assert memory['variants'][name]['library_sha256'] == final['artifacts'][name]
        assert memory['variants'][name]['owned_bytes'] == final['owned_bytes'][name]

    # Compare the entire new audio result with the saved, validated Oct3 output.
    # That earlier report links the fitted reference to the scored PCM files.
    old_cases = {item['case']: item for item in prior['cases']}
    assert len(old_cases) == len(audio['cases']) == 65
    assert set(old_cases) == {item['case'] for item in audio['cases']}
    for item in audio['cases']:
        old = old_cases[item['case']]
        assert item['input_sha256'] == old['input_sha256']
        assert item['samples'] == old['samples']
        for name in ('baseline', 'candidate'):
            assert item['pcm_sha256'][name] == old['pcm_sha256']['candidate']
            assert item['stream_sha256'][name] == old['stream_sha256']['candidate']

    validation = {}
    for kind in ('compatibility', 'fp_environment', 'block_alias', 'stride_boundary', 'stride_boundary_asan'):
        value, identity = record(f'oct7_{size}_{kind}.json')
        if kind == 'compatibility':
            assert value['artifacts'] == final['artifacts']
            assert all(item['output_and_state_bit_identical'] and item['frames'] == 1000
                       for item in value['parity'].values())
            assert all(item['serial_and_concurrent_bit_identical']
                       for item in value['concurrency'].values())
        else:
            assert value['passed']
        if kind == 'fp_environment':
            assert value['observed_process_thread_counts'] == [1]
            assert value['builds']['candidate0']['sha256'] == final['artifacts']['candidate']
        validation[kind] = identity

    safety = {}
    for suffix, count in (('', 5), ('_asan', 5), ('_scalar', 4)):
        path = ROOT/f'scratch/oct7/{size}/{SELECTED}{suffix}_contracts.log'
        content = path.read_text()
        assert f'100% tests passed, 0 tests failed out of {count}' in content
        build = ROOT/f'build/oct7_{size}_{SELECTED}{suffix}'
        safety[suffix or 'release'] = {
            'contracts_passed': count, 'log_sha256': sha(path),
            'library_sha256': sha(build/'libdpdf_full.so'), 'log': content,
        }

    screens = []
    for path in sorted((ROOT/'results').glob(f'oct7_{size}_*_screen.json')):
        value = json.loads(path.read_text())
        assert value['parity']['output_and_state_bit_identical']
        screens.append({'variant': value['variant'], 'path': str(path.relative_to(ROOT)),
                        'sha256': sha(path), 'owned_bytes': value['owned_bytes'],
                        'continuous': value['summary']['continuous']})
    standalone_pairs = []
    for repeat in range(4):
        pair = {name: item['implementations'][name]['wall']['mean_ms']
                for item in final['standalone_paced'] if item['repeat'] == repeat
                for name in item['implementations']}
        assert set(pair) == {'baseline', 'candidate'}
        standalone_pairs.append({'repeat': repeat, 'means_ms': pair,
                                 'mean_reduction_percent': 100*(1-pair['candidate']/pair['baseline'])})
    return {
        'selected': SELECTED, 'model_sha256': final['model_sha256'],
        'weights_sha256': final['weights_sha256'], 'library_sha256': final['artifacts'],
        'source_manifest': final['candidate_source_manifest'],
        'timing': final['summary'], 'owned_bytes': final['owned_bytes'],
        'standalone_pairs': standalone_pairs,
        'memory': {name: {key: value for key, value in item.items() if key != 'runs'}
                   for name, item in memory['variants'].items()},
        'audio': {'files': audio['selected_cases'], 'seconds': audio['total_audio_seconds'],
                  'hops': audio['total_audio_hops_including_flush'],
                  'all_spectra_states_and_pcm_exact': True,
                  'all_65_spectrum_state_and_pcm_hashes_match_saved_oct3_candidate': True,
                  'perceptual_scores_carried_forward_without_rescoring': True},
        'evidence': {'timing': final_id, 'audio': audio_id, 'memory': memory_id,
                     'saved_oct3_audio': prior_id, **validation},
        'safety': safety, 'screens': screens,
    }


def main():
    models = {str(size): collect(size) for size in (8, 2)}
    assembly, assembly_id = record('oct7_asm64_contract.json')
    assert assembly['passed']
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'summary_driver_sha256': sha(__file__),
        'reference': 'Accepted Oct3 best_single; separately matched for each model',
        'method': 'Four 1000-hop runs each continuous, paired 10 ms and standalone 10 ms; 100 warmup; no samples removed.',
        'limits': ['Linux x86-64 SysV assembly only; native Windows performance unmeasured',
                   'Inference calls exclude FFT, audio I/O and synthesis',
                   'Observed p99/max do not guarantee worst-case execution time',
                   'Saved perceptual scores cover six mixtures; 65 files verify exact output',
                   'Assembly memory operations are not instrumented by ASan; direct guard-page oracle is separate'],
        'models': models, 'assembly_contract': assembly_id,
    }
    output = ROOT/'results/oct7_summary.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'screened_builds': sum(len(m['screens']) for m in models.values()),
                      'standalone_time_reduction_percent': {size: item['timing']['standalone_paced']['mean_reduction_percent']
                                                            for size, item in models.items()}}))


if __name__ == '__main__':
    main()
