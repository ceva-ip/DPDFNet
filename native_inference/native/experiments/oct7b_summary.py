"""Verify retained Oct7b evidence and write its summary/report; no inference.

Run only after both selected models finish final timing, full exact audio,
compatibility, FP-control, release/sanitizer/scalar and fresh-process RSS checks.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re

from oct7b_optimization import COMBINATIONS


ROOT=Path(__file__).resolve().parents[2]
INITIAL=(
    'quant_fused64','quant_fused192','quant_fused192_init','quant_inline64',
    'quant_fused192_init_inline','compiler_ipo_nointerpose','compiler_pgo',
    'compiler_pgo_ipo_nointerpose','compiler_hot_align','graph_gru_fuse',
    'graph_bias_relu','graph_gru_approx','graph_activations_approx','dense_int8_pad8',
)
EXTRA=('conv_row_pair','conv_transpose_relu','quant_pair64','quant_pair64_96','quant_fused256')
DESCRIPTIONS={
    'quant_fused64':'Fuse K64 integer dot and FP32 epilogue',
    'quant_fused192':'Fuse complete K64/N192 row projection',
    'quant_fused192_init':'Above, initialize integer correction before dot',
    'quant_inline64':'Inline specialized K64 quantizer',
    'quant_fused192_init_inline':'Correction-first N192 fusion + inline quantizer',
    'compiler_ipo_nointerpose':'IPO + disable semantic interposition',
    'compiler_pgo':'Synthetic-trained PGO',
    'compiler_pgo_ipo_nointerpose':'PGO + IPO + disable interposition',
    'compiler_hot_align':'Align selected hot functions/loops',
    'graph_gru_fuse':'Fuse generated GRU with exact scalar libm gates',
    'graph_bias_relu':'Fuse bias and ReLU loops',
    'graph_gru_approx':'Generated GRU with approximate AVX2 gates',
    'graph_activations_approx':'Approximate generated Sigmoid/Tanh',
    'dense_int8_pad8':'Eight-column INT8 dense padding',
    'conv_row_pair':'Paired output rows in exact convolution',
    'conv_transpose_relu':'Fuse convolution transpose/ReLU',
    'quant_pair64':'Fixed K64 four-row/two-tile assembly',
    'quant_pair64_96':'Fixed K64/K96 paired assembly',
    'quant_fused256':'Fuse external-GRU K256/N768 projection',
}
QUALITY_METRICS=(
    ('raw.pesq_wb_16k','PESQ-WB (16 kHz resampling)'),
    ('raw.stoi','STOI'),('raw.si_snr_48k_db','SI-SNR (48 kHz, dB)'),
    ('raw.MOS_COL','SIGMOS coloration'),('raw.MOS_DISC','SIGMOS discontinuity'),
    ('raw.MOS_LOUD','SIGMOS loudness'),('raw.MOS_NOISE','SIGMOS noise'),
    ('raw.MOS_REVERB','SIGMOS reverberation'),('raw.MOS_SIG','SIGMOS signal'),
    ('raw.MOS_OVRL','SIGMOS overall'),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    path=Path(path)
    return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)}


def record(name):
    path=ROOT/'results'/name
    if not path.is_file():
        raise RuntimeError('Required evidence missing: '+str(path))
    return json.loads(path.read_text()),identity(path)


def manifest(folder):
    return {p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()
            and (p.suffix in ('.c','.h','.S') or p.name=='CMakeLists.txt')}


def screens(size,baseline_hash):
    values=[]
    for path in sorted((ROOT/'results').glob(f'oct7b_{size}_*_screen.json')):
        value=json.loads(path.read_text())
        assert value['model_size']==size and value['artifacts']['baseline']==baseline_hash
        assert value['parity']['frames']==1000
        if value['approximate']:
            assert value['parity']['finite'] and value['parity']['requires_new_quality_evaluation']
        else:
            assert value['parity']['output_and_state_bit_identical']
        assert value['timed_frames_per_run']==600 and value['warmup']==100
        assert len(value['continuous'])==3
        values.append({'variant':value['variant'],'approximate':value['approximate'],
                       'identity':identity(path),'driver_sha256':value['driver_sha256'],
                       'artifacts':value['artifacts'],'source_manifest':value['candidate_source_manifest'],
                       'environment':value['environment'],'owned_bytes':value['owned_bytes'],
                       'parity':value['parity'],'continuous':value['summary']['continuous'],
                       'raw_continuous_runs':value['continuous'],
                       'pgo_identity':value.get('pgo_identity')})
    observed={item['variant'] for item in values}
    assert set(INITIAL)<=observed, f'Model {size}: initial14 screens incomplete'
    return values


def collect(size,selected):
    final,final_id=record(f'oct7b_{size}_{selected}_final.json')
    audio,audio_id=record(f'oct7b_{size}_{selected}_audio.json')
    memory,memory_id=record(f'oct7b_{size}_{selected}_memory.json')
    prior,prior_id=record(f'oct7_{size}_combo_asm_norm_audio.json')
    old_summary,_=record('oct7_summary.json')
    assert not final['approximate'] and final['variant']==selected
    assert final['parity']['output_and_state_bit_identical'] and final['parity']['frames']==1000
    assert final['artifacts']['baseline']==old_summary['models'][str(size)]['library_sha256']['candidate']
    assert final['model_sha256']==prior['model_sha256'] and final['weights_sha256']==prior['weights_sha256']
    assert final['artifacts']==audio['library_sha256']==memory['library_sha256']
    assert audio['all_selected_cases_bit_identical'] and audio['full_available_suite']
    assert audio['selected_cases']==len(audio['cases'])==65
    assert memory['completed']
    assert prior['library_sha256']['candidate']==final['artifacts']['baseline']
    for report in (audio,memory):
        assert report['model_sha256']==final['model_sha256']
        assert report['weights_sha256']==final['weights_sha256']
    source=ROOT/f'scratch/oct7b/{size}/{selected}'
    assert manifest(source)==final['candidate_source_manifest']
    assert json.loads((source/'source_manifest.json').read_text())==manifest(source)
    candidate=ROOT/f'build/oct7b_{size}_{selected}/libdpdf_full.so'
    assert sha(candidate)==final['artifacts']['candidate']
    assert sha(ROOT/f'build/oct7_{size}_combo_asm_norm/libdpdf_full.so')==final['artifacts']['baseline']
    old_cases={item['case']:item for item in prior['cases']}
    assert set(old_cases)=={item['case'] for item in audio['cases']}
    for case in audio['cases']:
        old=old_cases[case['case']]
        assert case['input_sha256']==old['input_sha256'] and case['samples']==old['samples']
        assert case['every_output_and_state_bit_identical'] and case['pcm_bit_identical']
        assert case['all_values_finite'] and case['reset_replay_bit_identical']
        for name in ('baseline','candidate'):
            assert case['pcm_sha256'][name]==old['pcm_sha256']['candidate']
            assert case['stream_sha256'][name]==old['stream_sha256']['candidate']
    for mode in ('continuous','paced','standalone_paced'):
        expected=8 if mode=='standalone_paced' else 4
        assert len(final[mode])==expected
    assert final['timed_frames_per_run']==1000 and final['warmup']==100
    owned=final['owned_bytes']
    assert owned['baseline']-owned['candidate']==1440192
    for name in ('baseline','candidate'):
        assert memory['variants'][name]['owned_bytes']==owned[name]
        assert len(memory['variants'][name]['runs'])==4
    validation={}
    compatibility,compat_id=record(f'oct7b_{size}_compatibility.json')
    assert compatibility['artifacts']==final['artifacts']
    assert all(item['frames']==1000 and item['output_and_state_bit_identical']
               for item in compatibility['parity'].values())
    assert all(item['serial_and_concurrent_bit_identical'] for item in compatibility['concurrency'].values())
    fp,fp_id=record(f'oct7b_{size}_fp_environment.json')
    assert fp['passed'] and fp['observed_process_thread_counts']==[1]
    assert fp['builds']['baseline']['sha256']==final['artifacts']['baseline']
    assert fp['builds']['candidate0']['sha256']==final['artifacts']['candidate']
    assert len(fp['modes'])==8 and all(item['exact'] and item['finite'] for item in fp['modes'])
    contracts,contracts_id=record(f'oct7b_{size}_contracts.json')
    assert contracts['passed'] and contracts['variant']==selected and contracts['model_size']==size
    assert contracts['source_manifest']==final['candidate_source_manifest']
    assert {stage['suffix'] for stage in contracts['stages']}=={'','_asan','_scalar'}
    validation.update({'compatibility':compat_id,'fp_environment':fp_id,'contracts':contracts_id})
    safety={}
    for suffix in ('','_asan','_scalar'):
        log=ROOT/f'scratch/oct7b/{size}/{selected}{suffix}_contracts.log'
        text=log.read_text()
        result=re.search(r'100% tests passed, 0 tests failed out of (\d+)',text)
        assert result, str(log)
        count=int(result[1]); assert count>=4
        test_names=re.findall(r'Test\s+#\d+:\s+(\w+)\s+.*?Passed',text)
        assert len(test_names)==count
        library=ROOT/f'build/oct7b_{size}_{selected}{suffix}/libdpdf_full.so'
        safety[suffix or 'release']={'contracts_passed':count,'test_names':test_names,
                                   'log_identity':identity(log),'library_sha256':sha(library)}
        authoritative=next(stage for stage in contracts['stages'] if stage['suffix']==suffix)
        assert authoritative['returncode']==0 and authoritative['library_sha256']==sha(library)
        assert set(authoritative['registered_tests'])==set(test_names)
        assert all(oracle['checks']['passed'] for oracle in authoritative['direct_assembly_oracles'].values())
        safety[suffix or 'release']['direct_assembly_oracles']=authoritative['direct_assembly_oracles']
        if not suffix:
            assert sha(library)==final['artifacts']['candidate']
            for file,test in (('fused_contract.c','fused_contract'),
                              ('pair_contract.c','pair_contract'),('wide_contract.c','wide_contract')):
                if file in final['candidate_source_manifest']:
                    assert test in test_names
                    assert 'dpdf_'+test in authoritative['direct_assembly_oracles']
    # Additional standalone oracles, if root saved them, carry their raw report
    # identity. Added assembly CTest contracts above are always required when
    # the corresponding assembler is present in the selected source manifest.
    supplemental=[]
    for path in sorted((ROOT/'results').glob(f'oct7b_{size}_*.json')):
        if not any(token in path.stem for token in ('oracle','contract','boundary','alias','selected_dense')):
            continue
        value=json.loads(path.read_text())
        if value.get('passed'):
            supplemental.append({'identity':identity(path),'report':value})
    quality,quality_id=record(f'oct7b_{size}_graph_gru_approx_quality.json')
    assert quality['passed'] and quality['summary']['mixture']['clips']==6
    assert not quality['all_completed_pcm_bit_identical']
    assert quality['provenance']['builds']['baseline']['library_sha256']==final['artifacts']['baseline']
    assert quality['provenance']['model_sha256']==final['model_sha256']
    assert quality['provenance']['weights_sha256']==final['weights_sha256']
    approximate_quality={'accepted':False,'fixture_scope':'Six fixed EARS-WHAM mixtures, one per speaker',
                         'identity':quality_id,'summary':quality['summary']['mixture'],
                         'full_50_mixture_suite_completed':quality['full_50_mixture_suite_completed'],
                         'full_65_fixture_suite_completed':quality['full_65_fixture_suite_completed']}
    speech=None
    speech_path=ROOT/f'results/oct7b_{size}_speech_timing.json'
    if speech_path.is_file():
        value=json.loads(speech_path.read_text())
        assert value['variant']==selected and value['library_sha256']==final['artifacts']
        assert value['source_manifest']==final['candidate_source_manifest']
        assert value['model_sha256']==final['model_sha256'] and value['weights_sha256']==final['weights_sha256']
        assert value['parity']['frames']==1000 and value['parity']['output_and_state_bit_identical']
        assert len(value['runs'])==4 and value['sample_rate']==48000
        speech={'identity':identity(speech_path),'report':value}
    return {'selected':selected,'components':list(COMBINATIONS.get(selected,(selected,))),
            'environment':final['environment'],'model_sha256':final['model_sha256'],
            'weights_sha256':final['weights_sha256'],'library_sha256':final['artifacts'],
            'source_manifest':final['candidate_source_manifest'],'timing':final['summary'],
            'raw_timing_runs':{mode:final[mode] for mode in ('continuous','paced','standalone_paced')},
            'owned_bytes':owned,'owned_saving_bytes':1440192,
            'owned_reduction_percent':100*1440192/owned['baseline'],
            'memory':memory['variants'],'safety':safety,'supplemental_oracles':supplemental,
            'audio':{'files':65,'seconds':audio['total_audio_seconds'],
                     'hops':audio['total_audio_hops_including_flush'],
                     'all_spectra_states_and_pcm_exact':True,
                     'all65_hashes_match_saved_oct7_candidate':True,
                     'perceptual_scores_carried_forward_without_rescoring':True},
            'screens':screens(size,final['artifacts']['baseline']),
            'speech_timing':speech,'authoritative_contract_report':contracts,
            'approximate_gru_quality':approximate_quality,'pgo_identity':final.get('pgo_identity'),
            'evidence':{'final':final_id,'audio':audio_id,'memory':memory_id,
                        'saved_oct7_audio':prior_id,**validation},
            'historical_oct7_standalone_mean_ms':old_summary['models'][str(size)]['timing']['standalone_paced']['candidate']['mean_ms']}


def link(evidence):
    path=Path(evidence['path'])
    target='../'+path.as_posix()
    return f'[{path.name}]({target})'


def write_report(report):
    models=report['models']; selected={size:item['selected'] for size,item in models.items()}
    lookup={size:{s['variant']:s for s in m['screens']} for size,m in models.items()}
    def reduction(size,variant):
        row=lookup[size].get(variant)
        return f"{row['continuous']['mean_reduction_percent']:+.2f}%" if row else 'Not separately screened'
    lines=['# Further optimization against the committed Oct7 profile','',
        'Investigation: 2026-10-07. Both selected profiles retain byte-identical output '
        'and complete recurrent state across the saved 65-file suite per model. They '
        'reduce retained model allocations by **1,440,192 bytes each**. These are '
        'isolated research builds; consumer integration presets are unchanged.','',
        'The reference is the committed `combo_asm_norm` profile from '
        '[the preceding investigation](OCT7_OPTIMIZATION.md). Its historical '
        '**1.813 / 0.919 ms** standalone means are dated observations. New reductions '
        'below use fresh matched reference and candidate calls in this session. '
        'Cross-session subtraction does not estimate the measured benefit.','',
        '| Model | Historical Oct7 reference | Same reference, rerun | Selected new profile | Matched reduction |',
        '| --- | ---: | ---: | ---: | ---: |']
    for size in ('8','2'):
        m=models[size]; t=m['timing']['standalone_paced']
        lines.append(f"| DPDFNet-{size} | {m['historical_oct7_standalone_mean_ms']:.3f} ms | "
                     f"{t['baseline']['mean_ms']:.3f} ms | **{t['candidate']['mean_ms']:.3f} ms** | "
                     f"**{t['mean_reduction_percent']:.2f}%** |")
    lines += ['', '## Selected changes','',
              f"- DPDFNet-8: `{selected['8']}` — "+', '.join(f'`{c}`' for c in models['8']['components'])+'.',
              f"- DPDFNet-2: `{selected['2']}` — "+', '.join(f'`{c}`' for c in models['2']['components'])+'.','',
              'The integer row kernels fuse correction, conversion, the separately '
              'rounded FP32 scale product and the existing bias FMA. Fixed paired '
              'kernels reuse four activation broadcasts for two output tiles. '
              'Eight-column INT8 dense padding removes unneeded projections and '
              'output scratch. Exact generated-GRU fusion keeps scalar libm gates '
              'and original arithmetic; exact convolution helpers preserve tap '
              'and multiply/add order. PGO, where selected, is trained on procedural '
              'spectral streams in a separate process and adds no inference thread.','',
              'DPDFNet-8 selects PGO, correction-first small-projection fusion and '
              'the inlined K64 quantizer. DPDFNet-2 selects ordinary small-projection '
              'correction order, bias/ReLU fusion and the wider K256 external-GRU '
              'projection kernel. The saved combination screens document those '
              'model-specific choices; the final confirmation validates each '
              'selected composition as a whole.','',
              'Different component choices reflect independent screens for the two '
              'model sizes. Individual reductions cannot be added: fused kernels, '
              'compiler decisions, instruction footprint and memory traffic interact. '
              'Neighboring combinations have their own live reference runs; screens '
              'do not prove the globally optimal composition or independent value '
              'of every retained component. No new approximation is selected.','',
              '## Confirmed timing and tails','',
              '| Model | Execution | Reference → candidate mean | Reduction | p99 reference → candidate | Maximum reference → candidate |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    modes=(('continuous','Continuous'),('paced','Paired 10 ms cadence'),
           ('standalone_paced','Standalone 10 ms cadence'))
    for size in ('8','2'):
        for key,label in modes:
            t=models[size]['timing'][key]; a=t['baseline']; b=t['candidate']
            lines.append(f"| {size} | {label} | {a['mean_ms']:.3f} → {b['mean_ms']:.3f} ms | "
                         f"{t['mean_reduction_percent']:.2f}% | {a['p99_ms']:.3f} → {b['p99_ms']:.3f} ms | "
                         f"{a['max_ms']:.3f} → {b['max_ms']:.3f} ms |")
    lines += ['', 'Four 1,000-hop runs in each mode, with 100 warmup hops per run, '
              'produce 12,000 timed calls per implementation per model. Means and '
              'p99 are medians of run statistics; maxima are largest observed calls. '
              'No samples, outliers or scheduler stalls are filtered from the '
              'statistical aggregation. Saved JSON retains every per-run summary, '
              'context-switch count and late-call event; ordinary per-hop sample '
              'arrays are not stored. Paired order reverses '
              'each hop, standalone implementation order reverses between repeats.','',
              '| Model | Standalone process CPU, reference → candidate | Calls above 10 ms, reference / candidate | Paired late completions, reference / candidate | Standalone late completions, reference / candidate |',
              '| --- | ---: | ---: | ---: | ---: |']
    for size in ('8','2'):
        m=models[size]; t=m['timing']['standalone_paced']; p=m['timing']['paced']
        over={name:sum(m['timing'][mode][name]['over_10ms'] for mode,_ in modes) for name in ('baseline','candidate')}
        lines.append(f"| {size} | {t['baseline']['process_cpu_ms']:.3f} → {t['candidate']['process_cpu_ms']:.3f} ms | "
                     f"{over['baseline']} / {over['candidate']} | {p['baseline']['late_completions']} / {p['candidate']['late_completions']} | "
                     f"{t['baseline']['late_completions']} / {t['candidate']['late_completions']} |")
    lines += ['', 'Observed p99/maxima and deadline counts are measurements, not a '
              'hard worst-case guarantee. Completion after scheduled release also '
              'includes wake delay and, in paired mode, the other implementation. '
              'OS scheduling, CPU power states and host activity remain variable. '
              'No build, validation, scoring or memory jobs overlapped latency runs. '
              'FFT, synthesis, resampling, Python work outside the C call and audio '
              'device I/O are excluded. Windows/HushMic pipeline performance remains '
              'unmeasured. The environment is Intel i7-8700, GCC 12.2, Linux Docker/WSL2, '
              'one calling thread and unrestricted CPU affinity.','',
              *(['A separate saved real EARS-WHAM mixture (clip00033) confirmation '
                 'used four balanced 1,000-hop continuous runs per implementation '
                 'and 100 warmup hops. FFT preparation stayed outside the timed call.']
                if any(m['speech_timing'] for m in models.values()) else []),
              *[f"DPDFNet-{size}: {m['speech_timing']['report']['summary']['baseline']['mean_ms']:.3f} → "
                f"{m['speech_timing']['report']['summary']['candidate']['mean_ms']:.3f} ms "
                f"({m['speech_timing']['report']['mean_reduction_percent']:.2f}% reduction), "
                f"p99 {m['speech_timing']['report']['summary']['baseline']['p99_ms']:.3f} → "
                f"{m['speech_timing']['report']['summary']['candidate']['p99_ms']:.3f} ms, "
                f"maximum {m['speech_timing']['report']['summary']['baseline']['max_ms']:.3f} → "
                f"{m['speech_timing']['report']['summary']['candidate']['max_ms']:.3f} ms, "
                'with exact spectrum/state. '+link(m['speech_timing']['identity'])+'.'
                for size,m in models.items() if m['speech_timing']],
              'The DPDFNet-8 real-clip maximum increased despite lower mean/p99. '
              'This single-clip confirmation does not establish uniformly better '
              'peak latency or replace the broader cadence tests.',
              '',
              '## Memory','',
              '| Model | Native owned bytes, reference → candidate | Reduction | Warmed incremental RSS, reference → candidate |',
              '| --- | ---: | ---: | ---: |']
    for size in ('8','2'):
        m=models[size]; rss=m['memory']; a=rss['baseline']['median_incremental_rss_bytes']/1048576
        b=rss['candidate']['median_incremental_rss_bytes']/1048576
        lines.append(f"| {size} | {m['owned_bytes']['baseline']:,} → **{m['owned_bytes']['candidate']:,}** | "
                     f"**{m['owned_reduction_percent']:.2f}%** | {a:.3f} → {b:.3f} MiB |")
    lines += ['', 'RSS uses four balanced fresh processes per implementation, 120 '
              'warmup hops, unmapped source weights and subtraction of the common '
              'imported-runtime baseline. It includes library pages, buffers and '
              'allocator retention, and differs from owned model allocations and '
              'total application RAM. The owned reduction comes from dense padding '
              'and scratch removal; it is exactly 1,440,192 bytes for each model.','',
              '## Screened directions','',
              f"There are **{report['all_screened_builds']} model/build screens** in this round. "
              'The first table contains all 28 initial model/direction screens '
              '(14 directions × two model sizes). Each has a separate 1,000-hop '
              'correctness check and three balanced 600-hop continuous timing runs. '
              'Positive percentages mean lower inference time. Approximate rows '
              'measure numerical drift and require fresh quality evidence.','',
              '| Direction | DPDFNet-8 reduction | DPDFNet-2 reduction | Arithmetic |',
              '| --- | ---: | ---: | --- |']
    ranked=sorted(INITIAL,key=lambda v:-sum(lookup[s][v]['continuous']['mean_reduction_percent'] for s in ('8','2')))
    for v in ranked:
        approx=any(lookup[s][v]['approximate'] for s in ('8','2'))
        lines.append(f"| {DESCRIPTIONS[v]} (`{v}`) | {reduction('8',v)} | {reduction('2',v)} | "
                     f"{'Approximate; not accepted' if approx else 'Exact on 1,000 tested hops'} |")
    lines += ['', 'Additional focused screens follow. A blank experiment can '
              'still be present in a selected combination without an individual '
              'measurement; no separate gain is inferred.','',
              '| Direction | DPDFNet-8 reduction | DPDFNet-2 reduction |',
              '| --- | ---: | ---: |']
    for v in EXTRA:
        lines.append(f"| {DESCRIPTIONS[v]} (`{v}`) | {reduction('8',v)} | {reduction('2',v)} |")
    combo_names=sorted({v for s in ('8','2') for v in lookup[s] if v not in INITIAL+EXTRA})
    lines += ['', '| Combination | Components | DPDFNet-8 reduction | DPDFNet-2 reduction |',
              '| --- | --- | ---: | ---: |']
    for v in combo_names:
        components=', '.join(COMBINATIONS.get(v,(v,)))
        lines.append(f"| `{v}` | {components} | {reduction('8',v)} | {reduction('2',v)} |")
    lines += ['', 'Full per-run statistics, parity, memory counts, source manifests '
              'and library hashes for every screen are retained in '
              '[oct7b_summary.json](../results/oct7b_summary.json). Percentages '
              'within this table use each screen’s own contemporaneous reference; '
              'absolute times across separate screens do not rank close variants.','',
              '## Quality and exactness','']
    for size in ('8','2'):
        m=models[size]; a=m['audio']
        lines.append(f"DPDFNet-{size} passed **{a['files']} files / {a['seconds']:.3f} seconds / "
                     f"{a['hops']:,} hops**. Every spectrum, entire recurrent state and aligned FP32 PCM "
                     'sample matched the live Oct7 reference byte for byte, including signed zero. '
                     'All 65 stream/PCM hashes also match the saved Oct7 candidate with identical '
                     'input, model and weight hashes. '+link(m['evidence']['audio'])+'.')
        lines.append('')
    lines += ['The 65 files include 50 EARS-WHAM mixtures, ten mixtures with speech '
              'at −50 dBFS RMS, two clean controls and three independent 120-second '
              'pure-noise streams. Existing PESQ/STOI/SI-SNR/SIGMOS scores therefore '
              'carry forward unchanged on those scored waveforms. The quality '
              'comparison remains the existing six-mixture subset; exactness over '
              '65 files does not enlarge its perceptual scoring coverage or prove '
              'universal equivalence to ONNX/FP16/INT8.','',
              'The separate `graph_gru_approx` experiment replaces scalar generated '
              'GRU gates with fitted AVX2 gates and changes output. It is **not '
              'accepted**. These six-mixture candidate-minus-Oct7-reference deltas '
              'are exploratory; full 50-mixture scoring, quiet/clean/noise controls '
              'and listening evidence are still required before adoption.','',
              '| Metric | DPDFNet-8 paired mean delta | DPDFNet-2 paired mean delta |',
              '| --- | ---: | ---: |']
    for key,label in QUALITY_METRICS:
        columns=[]
        for size in ('8','2'):
            entry=models[size]['approximate_gru_quality']['summary']['metrics'][key]
            assert entry['paired_clips']==6
            columns.append(f"{entry['paired_mean_delta']:+.7f}")
        lines.append(f'| {label} | '+ ' | '.join(columns)+' |')
    lines += ['', 'SIGMOS and SI-SNR use native 48 kHz PCM. PESQ-WB explicitly '
              'resamples to 16 kHz; it is not a 48 kHz PESQ metric. Standard STOI '
              'accepts the 48 kHz input and internally resamples to 10 kHz. All '
              'seven SIGMOS dimensions, per-clip deltas, paired extrema and '
              'waveform differences are retained in '+
              ', '.join(link(models[s]['approximate_gru_quality']['identity']) for s in ('8','2'))+'.','',
              '## Safety evidence and artifact identities','',
              '| Model | Release contracts | ASan/UBSan contracts | Scalar-only contracts |',
              '| --- | ---: | ---: | ---: |']
    for size in ('8','2'):
        safety=models[size]['safety']
        lines.append(f"| {size} | {safety['release']['contracts_passed']} | {safety['_asan']['contracts_passed']} | "
                     f"{safety['_scalar']['contracts_passed']} |")
    lines += ['', 'Both selected binaries passed four precision compatibility '
              'configurations with 1,000 recurrent hops each, independent-context '
              'serial/concurrent replay, and all four rounding modes with denormal '
              'handling off/on after model creation. One observed OS thread was '
              'maintained during creation/process/destruction. Exception flags '
              'are not compared. Correctness workers own independent streams; '
              'they add no worker to one model’s inference.','',
              'New fused/paired/wide assembly, when present, has a dedicated '
              'direct scalar-oracle CTest with protected mappings, read-only '
              'inputs, unaligned buffers and canaries. ASan does not instrument '
              'handwritten assembly memory accesses; those direct checks and '
              'documented static bounds are separate evidence. Sanitizer/scalar '
              'PGO variants use plain builds; release-binary exactness is checked '
              'separately. Detailed test names, logs/hashes, source manifests, '
              'training/counter identities and supplemental oracles are retained '
              'in the JSON summary.','',
              '| Model | Library | SHA256 |', '| --- | --- | --- |']
    for size in ('8','2'):
        for name in ('baseline','candidate'):
            lines.append(f"| {size} | {name} | `{models[size]['library_sha256'][name]}` |")
    lines += ['', '## Reproduce offline','',
              'Use the existing `dpdfnet-native-dev` image, cached models/weights '
              'and saved audio fixtures. Preserve `scratch/oct7/{8|2}/combo_asm_norm` '
              'and `build/oct7_{8|2}_combo_asm_norm`: these are the reference sources '
              'and binary identities. All new builds use isolated snapshots. '
              'No download, tool installation or network access is required.','',
              'The commands below describe a full replay in a fresh, isolated '
              '`native_inference` workspace. First regenerate or preserve the '
              'Oct7 reference using the preceding report, with identical source '
              'and library hashes. The summary also requires the retained screen '
              'and approximate-quality evidence from this investigation. Existing '
              'frozen binaries and reports can instead be verified by the summary '
              'driver without rebuilding or retraining.','',
              'From `native_inference/` in PowerShell, define an offline runner:','',
              '```powershell',
              'function Invoke-Native([string]$Command) {',
              '  docker run --rm --network none -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 `',
              '    --mount "type=bind,source=${PWD},target=/bench" --workdir /bench `',
              '    --entrypoint sh dpdfnet-native-dev -c $Command',
              '  if ($LASTEXITCODE -ne 0) { throw "Native stage failed: $Command" }',
              '}', '```','',
              'On a fresh output set, invoke stages sequentially. Do not repeat '
              'the release PGO build against the retained counters:','', '```powershell']
    for size in ('8','2'):
        v=selected[size]
        for phase in ('build','asan','scalar'):
            lines.append(f"Invoke-Native 'python native/experiments/oct7b_optimization.py {phase} {v} --size {size}'")
        lines += [f"Invoke-Native 'python native/experiments/oct7b_optimization.py final {v} --size {size} --frames 1000 --repeats 4'",
                  f"Invoke-Native 'python native/experiments/oct7b_speech_timing.py {v} --size {size}'"]
        for stage in ('contracts','compatibility','fp','dense','audio'):
            lines.append(f"Invoke-Native 'python native/experiments/oct7b_validate.py {v} --size {size} --stage {stage}'")
        lines.append(f"Invoke-Native 'python native/experiments/oct7b_memory.py {v} --size {size}'")
    lines += [f"Invoke-Native 'python native/experiments/oct7b_summary.py --selected8 {selected['8']} --selected2 {selected['2']}'",
              '```','',
              'Build/validation/scoring/RSS stages must not overlap latency. '
              'PGO builds are immutable after training: repeating a release build '
              'against retained counters is rejected. The CLI has no variant '
              'alias or alternate output-root option; independent replay needs '
              'a fresh isolated workspace. Assembly targets Linux '
              'x86-64 SysV with AVX2/FMA; fallback sources are preserved elsewhere.','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected8',required=True); p.add_argument('--selected2',required=True)
    args=p.parse_args()
    models={str(size):collect(size,selected) for size,selected in ((8,args.selected8),(2,args.selected2))}
    report={'generated_at':datetime.now(timezone.utc).isoformat(),'summary_driver_sha256':sha(__file__),
            'reference':'Committed Oct7 combo_asm_norm; fresh independent matched comparisons per model',
            'method':'Four 1,000-hop runs each continuous/paired 10 ms/standalone 10 ms; 100 warmup hops; unfiltered aggregation.',
            'initial_screened_builds':28,'all_screened_builds':sum(len(m['screens']) for m in models.values()),
            'limits':['Linux x86-64 SysV; native Windows and consumer audio pipeline unmeasured',
                      'Observed timing tails do not guarantee worst-case latency',
                      'FFT/synthesis/audio-device work are outside timing',
                      '65 exact files do not enlarge six-mixture perceptual scoring coverage',
                      'Approximate GRU six-mixture evidence is not accepted for adoption',
                      'ASan does not instrument assembly; protected mappings and static bounds are separate'],
            'models':models}
    # Complete verification precedes publishing either document.
    document=write_report(report)
    output=ROOT/'results/oct7b_summary.json'; doc=ROOT/'native/OCT7B_OPTIMIZATION.md'
    output.write_text(json.dumps(report,indent=2)+'\n'); doc.write_text(document,encoding='utf-8')
    print(json.dumps({'summary':str(output),'report':str(doc),'selected':{s:m['selected'] for s,m in models.items()},
                      'screened_builds':report['all_screened_builds']}))


if __name__=='__main__': main()
