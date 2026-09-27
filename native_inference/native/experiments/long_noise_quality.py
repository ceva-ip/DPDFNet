"""Three 120-second, speech-free, synthetic noise streams; continuous state."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import lfilter
import fullband_quality as f
from probe import initial_state, synthesize

OUT=f.DATA/'long_noise'
SYSTEMS=('onnx','fp16','int8','w7a8')
SR=48000
N=120*SR


def db_rms(x):
    return float(10*np.log10(max(float(np.mean(np.asarray(x,dtype=np.float64)**2)),1e-30)))


def pcm_sha(x):
    return hashlib.sha256(x.astype('<f4').tobytes()).hexdigest()


def fixtures():
    rng=np.random.default_rng(20260927)
    white=rng.normal(size=N)
    frequencies=np.fft.rfftfreq(N,1/SR)
    spectrum=np.fft.rfft(rng.normal(size=N))
    spectrum/=np.sqrt(np.maximum(frequencies,20.))
    spectrum[0]=0
    pink=np.fft.irfft(spectrum,n=N)
    time=np.arange(N,dtype=np.float64)/SR
    hiss=lfilter([1.],[1.,-.93],rng.normal(size=N))
    hiss/=np.std(hiss)
    envelope=.15+.85*(.5+.5*np.sin(2*np.pi*time/17))**2
    mechanical=envelope*(.7*hiss+.4*np.sin(2*np.pi*60*time)+.15*np.sin(2*np.pi*120*time))
    # Seeded, irregular decaying noise bursts; no recorded sound or speech.
    for start in rng.integers(SR,N-SR,size=80):
        length=int(rng.integers(240,4800))
        mechanical[start:start+length]+=.5*rng.normal(size=length)*np.exp(-np.arange(length)/(length/5))
    records=[]
    for name,x in [('white',white),('pink',pink),('mechanical',mechanical)]:
        x-=x.mean()
        x*=10**(-25/20)/np.sqrt(np.mean(x*x))
        x=x.astype(np.float32)
        assert len(x)==N and np.isfinite(x).all() and np.max(np.abs(x))<1
        path=OUT/(name+'.wav')
        sf.write(path,x,SR,subtype='FLOAT')
        records.append({'name':name,'samples':N,'duration_seconds':120,'sample_rate':SR,
                        'rms_dbfs':db_rms(x),'peak_dbfs':float(20*np.log10(np.max(np.abs(x)))),
                        'pcm_sha256':pcm_sha(x)})
    return records


def init():
    global REF
    REF=f.session(f.MODEL)


def run(job):
    fixture,system,fingerprint=job
    target=OUT/(fixture['name']+'_'+system+'.json')
    if target.exists():
        cached=json.loads(target.read_text()); assert cached['fingerprint']==fingerprint
        return cached
    source=OUT/(fixture['name']+'.wav')
    x,sr=sf.read(source,dtype='float32')
    assert sr==SR and pcm_sha(x)==fixture['pcm_sha256']
    frames=f.audio_spectra(source)
    model=REF if system=='onnx' else f.ExtendedModel(REF,*( (3,16,7) if system=='fp16' else (4,8,7)),
                        build=f.W7 if system=='w7a8' else f.BASE,weights=f.WEIGHTS)
    state=initial_state(REF)
    outputs=[]; checkpoints=[]; max_state=0.
    try:
        for i,frame in enumerate(frames):
            enhanced,state=model.run(None,{'spec':frame,'state_in':state})
            assert np.isfinite(enhanced).all() and np.isfinite(state).all()
            max_state=max(max_state,float(np.max(np.abs(state))))
            outputs.append(enhanced)
            if (i+1)%1000==0:
                checkpoints.append({'input_seconds':(i+1)*.01,'state_max_abs':float(np.max(np.abs(state)))})
    finally:
        if system!='onnx':model.close()
    y=synthesize(outputs)[f.DELAY:f.DELAY+N]
    assert y.shape==x.shape and np.isfinite(y).all()
    path=OUT/(fixture['name']+'_'+system+'.wav')
    sf.write(path,y,SR,subtype='FLOAT')
    saved,rate=sf.read(path,dtype='float32')
    assert rate==SR and pcm_sha(saved)==pcm_sha(y)
    in20=np.sqrt(np.mean(x.astype(np.float64).reshape(-1,960)**2,axis=1))
    out20=np.sqrt(np.mean(y.astype(np.float64).reshape(-1,960)**2,axis=1))
    gains=20*np.log10(np.maximum(out20,1e-15)/np.maximum(in20,1e-15))
    seconds=[{'start_seconds':i,'input_rms_dbfs':db_rms(x[i*SR:(i+1)*SR]),
              'output_rms_dbfs':db_rms(y[i*SR:(i+1)*SR]),
              'attenuation_db':db_rms(x[i*SR:(i+1)*SR])-db_rms(y[i*SR:(i+1)*SR])} for i in range(120)]
    segments=[]
    for a,b in [(0,1),(1,30),(30,60),(60,90),(90,120)]:
        segments.append({'start_seconds':a,'end_seconds':b,'attenuation_db':db_rms(x[a*SR:b*SR])-db_rms(y[a*SR:b*SR]),
                         'output_rms_dbfs':db_rms(y[a*SR:b*SR])})
    result={'fixture':fixture['name'],'system':system,'fingerprint':fingerprint,
            'samples':N,'model_hops_including_flush':len(frames),'state_reset_count':1,
            'all_output_and_state_finite':True,'max_state_abs':max_state,'state_checkpoints':checkpoints,
            'input_rms_dbfs':db_rms(x),'output_rms_dbfs':db_rms(y),'attenuation_db':db_rms(x)-db_rms(y),
            'output_peak_dbfs':float(20*np.log10(max(np.max(np.abs(y)),1e-30))),
            'clipped_samples':int(np.count_nonzero(np.abs(y)>=1)),
            'worst_20ms_gain_db':float(gains.max()),'worst_20ms_start_seconds':float(np.argmax(gains)*.02),
            'frames_20ms_amplified_over_3db':int(np.count_nonzero(gains>3)),
            'output_pcm_sha256':pcm_sha(y),'segments':segments,'per_second':seconds}
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prior=json.loads((f.DATA/'evaluation/manifest.json').read_text())['provenance']
    for key,path in [('model',f.MODEL),('weights',f.WEIGHTS),('baseline_library',f.BASE/'libdpdf_full.so'),('w7a8_library',f.W7/'libdpdf_full.so')]:
        assert prior[key]==f.sha(path)
    inputs=fixtures()
    provenance={'script_sha256':f.sha(__file__),'prior_artifacts':prior,'seed':20260927,
                'inputs':inputs,'notes':'Synthetic speech-free noise, -25 dBFS RMS. Continuous state for each 120s file; common 2400-sample delay removal. Parallel quality test, not a latency benchmark.'}
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    (OUT/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    results=[]
    with ProcessPoolExecutor(6,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        futures=[pool.submit(run,(fixture,system,fingerprint)) for fixture in inputs for system in SYSTEMS]
        for future in as_completed(futures):
            result=future.result(); results.append(result)
            print(len(results),'/12',result['fixture'],result['system'],'attenuation',round(result['attenuation_db'],3),flush=True)
    results.sort(key=lambda r:(r['fixture'],SYSTEMS.index(r['system'])))
    (f.ROOT/'results/fullband_long_noise.json').write_text(json.dumps({'provenance':provenance,'results':results},indent=2,allow_nan=False)+'\n')
    print('Completed all 12 continuous two-minute runs',flush=True)


if __name__=='__main__':
    main()
