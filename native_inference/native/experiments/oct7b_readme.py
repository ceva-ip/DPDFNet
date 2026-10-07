"""Publish only the current overview after the complete Oct7b summary exists.

This source-only updater performs no inference or benchmarking. It verifies
retained evidence, frozen source/library identities and the expected README
revision before preparing a change. Run --dry-run to inspect the text without
writing; the normal invocation snapshots the original README before updating it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / 'README.md'
SUMMARY = ROOT / 'results/oct7b_summary.json'
EXPECTED_README_TEXT_SHA256 = '431d41633da420d09343a68e8ec580dc688c9694a9e409c5ffba5c6be7a09056'
SELECTED = {'8': 'combo_exact8', '2': 'combo_wide2'}
OVERVIEW8 = '**Current overview — `dpdfnet8_48khz_hr` (48 kHz).**'
OVERVIEW2 = '**Current overview — `dpdfnet2_48khz_hr` (48 kHz).**'
INTEGRATION = '**HushMic integration:**'
MIB = 1048576


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def manifest(folder):
    return {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()
            and (p.suffix in ('.c', '.h', '.S') or p.name == 'CMakeLists.txt')}


def evidence_path(identity):
    path = (ROOT / identity['path']).resolve()
    require(path == ROOT or ROOT in path.parents, 'Evidence path escapes native_inference')
    require(path.is_file() and sha(path) == identity['sha256'],
            'Retained evidence changed: ' + str(path))
    return path


def verified_summary():
    require(SUMMARY.is_file(), 'Complete results/oct7b_summary.json is required')
    summary_hash = sha(SUMMARY)
    report = json.loads(SUMMARY.read_text(encoding='utf-8'))
    old = json.loads((ROOT / 'results/oct7_summary.json').read_text(encoding='utf-8'))
    require(set(report['models']) == set(SELECTED), 'Both model summaries are required')
    require(report['all_screened_builds'] >= 50, 'The complete 50-build screen is required')
    require((ROOT / 'native/OCT7B_OPTIMIZATION.md').is_file(), 'Oct7b research report is required')
    for size, selected in SELECTED.items():
        model = report['models'][size]
        prior = old['models'][size]
        require(model['selected'] == selected, 'Unexpected selected profile for model ' + size)
        require(model['library_sha256']['baseline'] == prior['library_sha256']['candidate'],
                'Current reference is not the accepted Oct7 library')
        frozen = ROOT / f'scratch/oct7/{size}/combo_asm_norm'
        require(manifest(frozen) == prior['source_manifest'], 'Accepted Oct7 sources changed')
        source = ROOT / f'scratch/oct7b/{size}/{selected}'
        require(manifest(source) == model['source_manifest'], 'Selected sources changed')
        require(json.loads((source / 'source_manifest.json').read_text()) == model['source_manifest'],
                'Selected saved source manifest changed')
        for name, folder in (
                ('baseline', ROOT / f'build/oct7_{size}_combo_asm_norm'),
                ('candidate', ROOT / f'build/oct7b_{size}_{selected}')):
            require(sha(folder / 'libdpdf_full.so') == model['library_sha256'][name],
                    'Measured library changed: ' + str(folder))
            require(len(model['memory'][name]['runs']) == 4, 'Four fresh RSS workers are required')
            require(model['memory'][name]['owned_bytes'] == model['owned_bytes'][name],
                    'RSS worker allocation counts differ')
        require(model['owned_bytes']['baseline'] - model['owned_bytes']['candidate'] == 1440192,
                'Unexpected allocation saving')
        audio = model['audio']
        require(audio['files'] == 65 and audio['all_spectra_states_and_pcm_exact']
                and audio['all65_hashes_match_saved_oct7_candidate']
                and audio['perceptual_scores_carried_forward_without_rescoring'],
                'Complete 65-file byte-exact quality carry-forward proof is required')
        require(model['authoritative_contract_report']['passed'], 'Contract validation failed')
        for stage in ('release', '_asan', '_scalar'):
            safety = model['safety'][stage]
            require(safety['contracts_passed'] >= 4, 'Complete safety contracts are required')
            evidence_path(safety['log_identity'])
            for oracle in safety['direct_assembly_oracles'].values():
                require(oracle['checks']['passed'], 'Direct assembly oracle failed')
        for identity in model['evidence'].values():
            evidence_path(identity)
        for mode, runs in model['raw_timing_runs'].items():
            require(len(runs) == (8 if mode == 'standalone_paced' else 4),
                    'Four final matched timing runs are required')
            for name in ('baseline', 'candidate'):
                values = model['timing'][mode][name]
                require(values['over_10ms'] == 0, 'An inference call exceeded 10 ms')
                if mode != 'continuous':
                    require(values['late_completions'] == 0, 'A cadence completion was late')
        require(model['model_sha256'] == prior['model_sha256']
                and model['weights_sha256'] == prior['weights_sha256'],
                'Model/weights differ from the historical quality reference')
    require(sha(SUMMARY) == summary_hash, 'Summary changed during verification')
    return report, summary_hash


def replace_once(text, old, new):
    require(text.count(old) == 1, 'Expected one README anchor: ' + old[:100])
    return text.replace(old, new, 1)


def quality_table(section):
    start = section.index('| Version | PESQ-WB')
    return section[start:section.index('\n\n', start)]


def retained_rows(text):
    prefixes = ('| Original ONNX FP32 |', '| Native selective FP16 |',
                '| Native selective INT8 |', '| Oct3 exact W7A8 (historical session) |')
    return [line for line in text.splitlines() if line.startswith(prefixes)]


def overview(section, model, size):
    """Retain the historical tables/quality values; replace overview prose only."""
    table_start = section.index('| Version | Inference / hop')
    table_end = section.index('\n\n', table_start)
    table = section[table_start:table_end]
    old_latest = next(line for line in table.splitlines() if line.startswith('| Latest Oct7 exact W7A8'))
    historical = old_latest.replace('Latest Oct7 exact W7A8 (single thread)',
                                    'Oct7 exact W7A8 (historical session)').replace('**', '')
    timing = model['timing']['standalone_paced']
    baseline, candidate = timing['baseline'], timing['candidate']
    rss = model['memory']['candidate']['median_incremental_rss_bytes'] / MIB
    owned = model['owned_bytes']['candidate'] / MIB
    latest = (f'| Latest Oct7b exact W7A8 (single thread) | **{candidate["mean_ms"]:.3f} ms** | '
              f'**{rss:.2f} MiB** | **{owned:.2f} MiB** |')
    table = replace_once(table, old_latest, historical + '\n' + latest)
    quality = section[section.index('**Output quality**'):].rstrip()
    audio_path = model['evidence']['audio']['path'].replace('\\', '/')
    quality = replace_once(quality, f'results/oct7_{size}_combo_asm_norm_audio.json', audio_path)
    if size == '8':
        quality = replace_once(quality,
            'The latest exact kernels preserve the fitted reference byte for byte over',
            'The latest exact kernels preserve the Oct7 and fitted references byte for byte over')
        intro = (OVERVIEW8 + ' The fitted\n'
            'W7A8 reference is **W7A8 + pack32 + fitted degree-5 GRU gates**\n'
            '(`build/w7_followup_pack_fit5`). FP16 and INT8 below are the selective native\n'
            'precision presets; W7A8 uses 7-bit weights and 8-bit activations in its quantized\n'
            'kernels. The latest exact research profile is `build/oct7b_8_combo_exact8`;\n'
            'the production INT8 preset remains unchanged.\n\n'
            '**Latency and memory footprint** — Intel i7-8700, one inference thread,\n'
            'Linux Docker/WSL2, 10 ms audio hops:\n\n')
        methods = (
            'The first three rows use the [2026-09-21 matched comparison](native/ONNX_LATEST_COMPARISON.md)\n'
            '(median of four run means, including Python call overhead and output allocation).\n'
            'The Oct3 and earlier Oct7 rows retain their [October 3](native/FURTHER_EXACT_OPTIMIZATION.md)\n'
            'and [first October 7](native/OCT7_OPTIMIZATION.md) historical measurements.\n'
            'The latest row uses the [next October 7 matched comparison](native/OCT7B_OPTIMIZATION.md):\n'
            'median of four standalone cadence run means, preallocated C calls, 100 warmup\n'
            'and 1,000 timed hops per run. '
            f'**The same Oct7 library reran at {baseline["mean_ms"]:.3f} ms**, versus\n'
            f'**{candidate["mean_ms"]:.3f} ms** for the selected profile: '
            f'**{timing["mean_reduction_percent"]:.2f}% less time**.\n'
            'Its historical 1.813 ms and fresh 1.907 ms describe identical binaries in\n'
            'different sessions. Likewise, Oct3’s historical 1.867 ms and its 1.963 ms\n'
            'rerun in the earlier Oct7 study describe identical binaries. The cause of\n'
            'session variation was not isolated. Direct speedups against historical\n'
            'rows would mix sessions or call methods. FFT, audio I/O and resampling\n'
            'are excluded; the model’s **50 ms algorithmic delay is unchanged**.\n')
    else:
        quality = replace_once(quality, 'native/OCT7_OPTIMIZATION.md', 'native/OCT7B_OPTIMIZATION.md')
        intro = (OVERVIEW2 + ' The selected exact Oct7b W7A8\n'
            'profile is `build/oct7b_2_combo_wide2`, with the same fitted reference and\n'
            'unchanged weights. It remains an experimental candidate, with the\n'
            'production INT8 preset unchanged.\n\n'
            '**Latency and memory footprint** — same CPU, single-thread execution and\n'
            '10 ms cadence as above:\n\n')
        methods = (
            'The first three latency rows retain the [2026-09-27 matched comparison](results/dpdfnet2_overview_timing.json):\n'
            'median of four run means, 100 warmup + 1,000 timed hops per run, rotating order,\n'
            'including Python overhead and output allocation. The Oct3 and earlier Oct7\n'
            'rows retain their [October 3](results/further2_best_single_final.json) and\n'
            '[first October 7](results/oct7_2_combo_asm_norm_final.json) historical measurements.\n'
            'The latest row uses the [next October 7 matched preallocated C comparison](native/OCT7B_OPTIMIZATION.md),\n'
            'with the same four-run standalone cadence method as the latest DPDFNet-8 row.\n'
            f'**The same Oct7 library reran at {baseline["mean_ms"]:.3f} ms**, versus\n'
            f'**{candidate["mean_ms"]:.3f} ms** for the selected profile: '
            f'**{timing["mean_reduction_percent"]:.2f}% less time**.\n'
            'Its historical 0.919 ms and fresh 0.942 ms describe identical binaries in\n'
            'different sessions. Direct speedups against historical rows would mix\n'
            'sessions or methods. Cross-model ratios remain approximate because the\n'
            'models were measured in separate sessions. The **50 ms algorithmic delay\n'
            'is unchanged**; FFT, audio I/O, synthesis and resampling are excluded.\n')
    tail = (f'Standalone p99 fell **{baseline["p99_ms"]:.3f} → {candidate["p99_ms"]:.3f} ms**. '
            f'Maximum {"fell" if candidate["max_ms"] < baseline["max_ms"] else "increased"} '
            f'**{baseline["max_ms"]:.3f} → {candidate["max_ms"]:.3f} ms**'
            + ('.\n' if size == '8' else ', so peak latency did not improve uniformly.\n')
            + 'Both builds had zero calls above 10 ms and zero late completions across\n'
            '12,000 timed calls each. Observed tails do not guarantee worst-case\n'
            'latency; see the [new report](native/OCT7B_OPTIMIZATION.md) and\n'
            '[tail investigation](native/TAIL_LATENCY_INVESTIGATION.md).\n')
    memory_path = model['evidence']['memory']['path'].replace('\\', '/')
    memory = (
        'RSS is warmed process memory above the common imported-runtime baseline,\n'
        'including allocator retention and stream buffers; it is not total application\n'
        'RAM or model file size. Native owned allocations count memory owned by the C\n'
        'model and are **not interchangeable with RSS**. '
        f'The selected profile saves\n**{model["owned_saving_bytes"]:,} owned bytes '
        f'({model["owned_reduction_percent"]:.2f}%)** relative to the unchanged Oct7 reference,\n'
        'by avoiding padded INT8 columns in narrow dense layers. The\n'
        f'[new RSS measurement]({memory_path}) uses the same protocol:\n'
        'median of four fresh processes per build, each with 120 warmup hops and\n'
        'source weights unmapped before sampling. RSS includes allocator and code\n'
        'effects, so its measured reduction differs from exact owned-byte savings.\n')
    if size == '2':
        memory = ('ONNX/FP16/INT8 RSS comes from the [original matched memory study](results/onnx_latest_summary.json).\n'
                  + memory)
    return intro + table + '\n\n' + methods + tail + '\n' + memory + '\n' + quality + '\n\n'


def updated_text(original, report):
    start8 = original.index(OVERVIEW8)
    start2 = original.index(OVERVIEW2)
    suffix_start = original.index(INTEGRATION)
    model8, model2 = report['models']['8'], report['models']['2']
    t8, t2 = (m['timing']['standalone_paced'] for m in (model8, model2))
    require(model8['audio']['seconds'] == model2['audio']['seconds']
            and model8['audio']['hops'] == model2['audio']['hops'],
            'The two complete saved fixture suites differ')
    heading = (
        '# DPDFNet native inference investigation\n\n'
        '**2026-10-07 next single-thread W7A8 optimization:** the\n'
        '[validated follow-up research](native/OCT7B_OPTIMIZATION.md) combines fused\n'
        'quantized assembly, exact graph/convolution changes and smaller retained\n'
        f'INT8 storage after **{report["all_screened_builds"]} screened builds**. In fresh matched sessions,\n'
        f'standalone 10 ms cadence inference falls **{t8["baseline"]["mean_ms"]:.3f} → '
        f'{t8["candidate"]["mean_ms"]:.3f} ms ({t8["mean_reduction_percent"]:.2f}% less)** for `dpdfnet8_48khz_hr`\n'
        f'and **{t2["baseline"]["mean_ms"]:.3f} → {t2["candidate"]["mean_ms"]:.3f} ms '
        f'({t2["mean_reduction_percent"]:.2f}% less)** for `dpdfnet2_48khz_hr`, relative to\n'
        'the unchanged accepted Oct7 builds rerun alongside the candidates. Their\n'
        'earlier **1.813 / 0.919 ms** measurements remain historical results from\n'
        'another session. Percentages compare fresh matched runs; the tables retain\n'
        'both historical profiles and the latest results.\n\n'
        'All output spectra, complete recurrent states and PCM remain byte-identical\n'
        f'across **65 files / {model8["audio"]["seconds"] / 60:.1f} minutes / '
        f'{model8["audio"]["hops"]:,} hops per model**. The six-mixture\n'
        'PESQ/STOI/SI-SNR/SIGMOS scores therefore carry forward unchanged. Each\n'
        f'profile saves **1,440,192 owned bytes** (**{model8["owned_reduction_percent"]:.2f}% / '
        f'{model2["owned_reduction_percent"]:.2f}%**) with one inference thread. Standalone p99\n'
        'improves for both models; the DPDFNet-2 maximum increases. Neither build\n'
        'has a call above 10 ms or a late cadence completion in the retained final runs.\n'
        'The report includes measured RSS, timing tails, safety checks and reproduction.\n'
        'Assembly targets Linux x86-64 SysV; native Windows timing is unmeasured.\n'
        'These are isolated research builds; distributed integration presets remain unchanged.\n\n')
    new8 = overview(original[start8:start2], model8, '8')
    new2 = overview(original[start2:suffix_start], model2, '2')
    updated = heading + new8 + new2 + original[suffix_start:]
    require(retained_rows(updated) == retained_rows(original), 'Historical rows or quality values changed')
    require(quality_table(new8) == quality_table(original[start8:start2]), 'Model8 quality table changed')
    require(quality_table(new2) == quality_table(original[start2:suffix_start]), 'Model2 quality table changed')
    require(updated[updated.index(INTEGRATION):] == original[suffix_start:], 'Unrelated README sections changed')
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Print proposed README; do not write or snapshot')
    args = parser.parse_args()
    original_bytes = README.read_bytes()
    original = original_bytes.decode('utf-8').replace('\r\n', '\n')
    require(text_sha(original) == EXPECTED_README_TEXT_SHA256,
            'README changed since this updater was prepared; review its anchors before applying')
    report, summary_hash = verified_summary()
    updated = updated_text(original, report)
    require(README.read_bytes() == original_bytes and sha(SUMMARY) == summary_hash,
            'README or verified summary changed while preparing the update')
    if args.dry_run:
        sys.stdout.reconfigure(encoding='utf-8')
        print(updated, end='')
        return
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    backup = ROOT / f'scratch/oct7b/readme/{original_hash}.md'
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        require(backup.read_bytes() == original_bytes, 'Existing README snapshot differs')
    else:
        backup.write_bytes(original_bytes)
    newline = '\r\n' if b'\r\n' in original_bytes else '\n'
    payload = updated.replace('\n', newline).encode('utf-8')
    require(README.read_bytes() == original_bytes and sha(SUMMARY) == summary_hash,
            'README or summary changed before publication')
    README.write_bytes(payload)
    print(json.dumps({'readme': str(README), 'original_sha256': original_hash,
                      'updated_sha256': sha(README), 'summary_sha256': summary_hash,
                      'original_snapshot': str(backup)}))


if __name__ == '__main__':
    main()
