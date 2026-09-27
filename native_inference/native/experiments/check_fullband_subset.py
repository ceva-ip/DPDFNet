"""Check subset replay against unmodified test generation on a small corpus.

Exercises preceding-cut noise reuse, new noise draws, ramps, highpass filtering,
and clipping adjustment. Uses deterministic synthetic files, not benchmark scores.
"""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import prepare_fullband_subset as subset


def main():
    original_root=subset.ROOT
    root=original_root/'replay_check'
    root.mkdir(exist_ok=True)
    source=(original_root/'generate_ears_wham.py').read_text()
    (root/'generate_ears_wham.py').write_text(source)
    rng=np.random.default_rng(1234)
    cuts={}
    meta={}
    for speaker in range(102,108):
        s=f'p{speaker}'
        cuts[s]={}
        directory=root/'data/EARS'/s
        directory.mkdir(parents=True,exist_ok=True)
        for name in ('emo_anger_sentences','emo_serenity_sentences'):
            length=11*48000
            audio=(.32*np.sin(np.arange(length)*(.013+.001*(speaker-102)))+.015*rng.normal(size=length)).astype(np.float64)
            path=directory/(name+'.wav')
            sf.write(path,audio,48000,subtype='FLOAT')
            meta[str(path)]={'frames':length,'channels':1,'sample_rate':48000}
            cuts[s][name]=[[0,4*48000],[4*48000,7*48000],[7*48000,11*48000]]
    (root/'test_files.json').write_text(json.dumps(cuts))
    noise=root/'data/WHAM48kHz/high_res_wham'
    (noise/'audio').mkdir(parents=True,exist_ok=True)
    names=[]
    for i in range(3):
        path=noise/'audio'/f'noise{i}.wav'
        length=(5+i)*48000
        sf.write(path,rng.normal(0,.1,(length,2)),48000,subtype='FLOAT')
        meta[str(path)]={'frames':length,'channels':2,'sample_rate':48000}
        names.append(path.name)
    (noise/'high_res_metadata.csv').write_text('Filename,WHAM! Split\n'+''.join(f'{n},Test\n' for n in names))
    namespace={'__name__':'test_generator'}
    source=source.replace('for subset in ["train", "valid"]:','for subset in []:')
    exec(compile(source,'official_ears_generator','exec'),namespace)
    namespace['exists']=lambda path:False
    namespace['makedirs']=lambda path:os.makedirs(path,exist_ok=True)
    args=SimpleNamespace(data_dir=str(root/'data'),sr=48000,min_snr=-2.5,max_snr=17.5,min_length=4.,cut_length=10.,
                         cutoff_freq=75.,min_dB=-55.,ramp_time_in_ms=10,max_time_test_set_in_s=29)
    setattr(args,'16k',False)
    os.chdir(root)
    namespace['main'](args)
    baseline_rng=np.random.get_state()
    subset.ROOT=root
    selection=subset.select(meta,12)
    chosen={e['id'] for e in selection['clips']}
    test=root/'data/EARS-WHAM_v2'
    def pcm_hashes():
        return {str(p.relative_to(test)):hashlib.sha256(sf.read(p,dtype='float32')[0].tobytes()).hexdigest()
                for p in test.glob('test/*/*/*.wav') if p.stem.split('_')[0] in chosen}
    before=pcm_hashes()
    rows=(test/'test.csv').read_text().splitlines()
    expected_rows=[r for r in rows[1:] if r.split(',')[0] in chosen]
    subset.generate(meta,selection,False)
    after=pcm_hashes()
    assert before==after and len(before)==24
    assert expected_rows==(test/'test.csv').read_text().splitlines()[1:]
    actual_rng=np.random.get_state()
    assert baseline_rng[0]==actual_rng[0] and np.array_equal(baseline_rng[1],actual_rng[1]) and baseline_rng[2:]==actual_rng[2:]
    report={'selected_clips':12,'matched_clean_and_noisy_waveforms':24,'exact_pcm_match':True,
            'exact_selected_csv_rows':True,'exact_final_rng_state':True,
            'check_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (original_root/'subset_replay_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
