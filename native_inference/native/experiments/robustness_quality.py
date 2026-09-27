"""Ten low-level EARS mixtures and two clean controls, all processed at 48 kHz."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pesq import pesq
from pystoi import stoi
import fullband_quality as f

OUT = f.DATA/'robustness'
RESULT = f.ROOT/'results/fullband_robustness.json'


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))


def level(x):
    return float(20*np.log10(max(rms(x),1e-30)))


def metrics(clean, audio):
    result={'si_snr_48k_db':f.si_snr(clean,audio), **f.SCORER.run(audio,sr=48000)}
    for key,call in [('pesq_wb_16k',lambda:pesq(16000,resample_poly(clean,1,3),resample_poly(audio,1,3),'wb')),
                     ('stoi',lambda:stoi(clean,audio,48000))]:
        try:
            result[key]=float(call())
        except Exception as error:
            result[key]=None
            result[key+'_error']=str(error)
    return result


def preservation(clean,audio):
    c=np.asarray(clean,dtype=np.float64); y=np.asarray(audio,dtype=np.float64)
    alpha=float(np.dot(c,y)/np.dot(c,c))
    n=len(c)//960
    cr=np.sqrt(np.mean(c[:n*960].reshape(n,960)**2,axis=1))
    yr=np.sqrt(np.mean(y[:n*960].reshape(n,960)**2,axis=1))
    active=cr>=cr.max()*.1
    return {'output_rms_dbfs':level(y),'output_peak_dbfs':float(20*np.log10(max(np.max(np.abs(y)),1e-30))),
            'rms_gain_vs_clean_db':level(y)-level(c),'projection_gain':alpha,
            'projection_gain_db':float(20*np.log10(max(abs(alpha),1e-30))),
            'ordinary_snr_vs_clean_db':float(10*np.log10(np.dot(c,c)/max(np.dot(y-c,y-c),1e-30))),
            'active_frames_suppressed_over_20db_fraction':float(np.mean(yr[active]<cr[active]*.1))}


def run(job):
    case,fingerprint=job
    file=OUT/(case['case_id']+'.json')
    if file.exists():
        result=json.loads(file.read_text()); assert result['fingerprint']==fingerprint
        return result
    e=case['clip']
    clean,sr=sf.read(f.DATA/e['clean'],dtype='float32')
    source,nsr=sf.read(f.DATA/(e['noisy'] if case['scenario']=='low_level' else e['clean']),dtype='float32')
    assert sr==nsr==48000 and clean.shape==source.shape and clean.ndim==1
    gain=10**(-50/20)/rms(clean) if case['scenario']=='low_level' else 1.
    target=(clean*gain).astype(np.float32); inp=(source*gain).astype(np.float32)
    directory=OUT/'audio'/case['case_id']; directory.mkdir(parents=True,exist_ok=True)
    sf.write(directory/'input.wav',inp,48000,subtype='FLOAT')
    sf.write(directory/'clean.wav',target,48000,subtype='FLOAT')
    if case['scenario']=='low_level':
        assert abs(level(target)+50)<1e-5
    frames=f.audio_spectra(directory/'input.wav')
    result={'case':case,'fingerprint':fingerprint,'samples':len(clean),'gain':gain,
            'source_clean_sha256':f.sha(f.DATA/e['clean']), 'source_input_sha256':f.sha(f.DATA/(e['noisy'] if case['scenario']=='low_level' else e['clean'])),
            'original_clean_rms_dbfs':level(clean),'test_clean_rms_dbfs':level(target),
            'test_input_rms_dbfs':level(inp),'systems':{}}
    for system in ('input','onnx','fp16','int8','w7a8'):
        audio=inp if system=='input' else f.enhance(f.MODELS[system],frames,f.REFERENCE)[f.DELAY:f.DELAY+len(inp)]
        assert audio.shape==target.shape and np.isfinite(audio).all()
        sf.write(directory/(system+'.wav'),audio,48000,subtype='FLOAT')
        entry={'raw':metrics(target,audio),'preservation':preservation(target,audio),
               'pcm_sha256':hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest()}
        if case['scenario']=='clean' and system=='input':
            entry['raw']['si_snr_48k_db']=None  # Identical signals: mathematically infinite.
            entry['preservation']['ordinary_snr_vs_clean_db']=None
        if case['scenario']=='low_level':
            restored=(audio/gain).astype(np.float32)
            entry['level_restored']=metrics(clean,restored)
            if system!='input':
                normal,rate=sf.read(f.DATA/'evaluation/audio'/system/e['speaker']/(e['id']+'.wav'),dtype='float32')
                assert rate==48000 and normal.shape==restored.shape
                entry['scale_equivariance_snr_db']=float(10*np.log10(np.sum(normal.astype(np.float64)**2)/max(np.sum((normal.astype(np.float64)-restored)**2),1e-30)))
        result['systems'][system]=entry
    file.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    original=json.loads((f.DATA/'evaluation/manifest.json').read_text())
    for key,path in [('model',f.MODEL),('weights',f.WEIGHTS),('baseline_library',f.BASE/'libdpdf_full.so'),('w7a8_library',f.W7/'libdpdf_full.so')]:
        assert f.sha(path)==original['provenance'][key]
    # Selection uses speaker balance and a fixed hash, never previous scores.
    pool=list(original['clips']); counts={e['speaker']:0 for e in pool}; selected=[]
    for _ in range(10):
        e=min(pool,key=lambda e:(counts[e['speaker']],hashlib.sha256(('low50:'+e['id']).encode()).hexdigest()))
        selected.append(e); counts[e['speaker']]+=1; pool.remove(e)
    regular=next(e for e in original['clips'] if 'regular' in e['speech_file'])
    freeform=next(e for e in original['clips'] if 'freeform_speech' in e['speech_file'] and e['speaker']!=regular['speaker'])
    cases=[{'case_id':'low_'+e['id'],'scenario':'low_level','clip':e} for e in selected]
    cases += [{'case_id':'clean_'+e['id'],'scenario':'clean','clip':e} for e in (regular,freeform)]
    provenance={'prior_run':original['provenance'],'script_sha256':f.sha(__file__),
                'low_level_definition':'Clean full-clip RMS -50 dBFS; identical gain on speech and mixture preserves SNR',
                'raw_scores':'Unmodified quiet output; no automatic gain compensation before the model or SIGMOS',
                'level_restored_scores':'Common inverse input gain applied after enhancement only, to isolate artifacts from absolute loudness',
                'selection':'Greedy speaker balance with fixed SHA256 ordering; clean: first regular and different-speaker freeform',
                'cases':cases}
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    results=[]
    with ProcessPoolExecutor(6,mp_context=multiprocessing.get_context('spawn'),initializer=f.init_worker) as pool:
        for future in as_completed([pool.submit(run,(c,fingerprint)) for c in cases]):
            result=future.result(); results.append(result)
            print(len(results),'/ 12',result['case']['case_id'],flush=True)
    results.sort(key=lambda r:r['case']['case_id'])
    summary={}
    for scenario in ('low_level','clean'):
        group=[r for r in results if r['case']['scenario']==scenario]
        summary[scenario]={}
        for system in ('input','onnx','fp16','int8','w7a8'):
            entry={}
            for category in ('raw','preservation')+(('level_restored',) if scenario=='low_level' else ()):
                entry[category]={k:float(np.mean([r['systems'][system][category][k] for r in group]))
                                 for k,v in group[0]['systems'][system][category].items() if isinstance(v,(int,float)) and all(isinstance(r['systems'][system][category].get(k),(int,float)) for r in group)}
            if scenario=='low_level' and system!='input':
                normal=[json.loads((f.DATA/'evaluation'/(r['case']['clip']['id']+'.json')).read_text())['systems'][system] for r in group]
                entry['normal_level_means']={k:float(np.mean([r[k] for r in normal])) for k in normal[0] if isinstance(normal[0][k],(int,float))}
                entry['low_minus_normal']={k:entry['level_restored'][k]-v for k,v in entry['normal_level_means'].items()}
                entry['mean_scale_equivariance_snr_db']=float(np.mean([r['systems'][system]['scale_equivariance_snr_db'] for r in group]))
            summary[scenario][system]=entry
    RESULT.write_text(json.dumps({'provenance':provenance,'summary':summary,'cases':results},indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
