"""Verify scored PCM and inspect paired level sensitivity and suppression flags."""
import json
import hashlib
import numpy as np
import soundfile as sf
import fullband_quality as f
from robustness_quality import OUT, RESULT, preservation

report=json.loads(RESULT.read_text())
rows=[]; verified=0
for case in report['cases']:
    cid=case['case']['case_id']; e=case['case']['clip']; directory=OUT/'audio'/cid
    clean,_=sf.read(directory/'clean.wav',dtype='float32')
    row={'case_id':cid,'original_clean_rms_dbfs':case['original_clean_rms_dbfs'],'systems':{}}
    old=json.loads((f.DATA/'evaluation'/(e['id']+'.json')).read_text())
    original_clean,_=sf.read(f.DATA/e['clean'],dtype='float32')
    for system,entry in case['systems'].items():
        audio,rate=sf.read(directory/(system+'.wav'),dtype='float32')
        assert rate==48000 and audio.shape==clean.shape and np.isfinite(audio).all()
        assert hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest()==entry['pcm_sha256']
        assert not any(k.endswith('_error') for k in entry['raw'])
        verified+=1
        if system=='input': continue
        n=len(clean)//960
        cr=np.sqrt(np.mean(clean[:n*960].astype(np.float64).reshape(n,960)**2,axis=1))
        yr=np.sqrt(np.mean(audio[:n*960].astype(np.float64).reshape(n,960)**2,axis=1))
        flags=(cr>=cr.max()*.1)&(yr<cr*.1)
        item={'suppressed_frame_start_seconds':(np.flatnonzero(flags)*.02).tolist(),
              'projection_gain_db':entry['preservation']['projection_gain_db']}
        if case['case']['scenario']=='low_level':
            normal,_=sf.read(f.DATA/'evaluation/audio'/system/e['speaker']/(e['id']+'.wav'),dtype='float32')
            item['low_minus_normal_si_snr_db']=entry['raw']['si_snr_48k_db']-old['systems'][system]['si_snr_48k_db']
            item['low_minus_normal_pesq']=entry['level_restored']['pesq_wb_16k']-old['systems'][system]['pesq_wb_16k']
            item['normal_suppressed_fraction']=preservation(original_clean,normal)['active_frames_suppressed_over_20db_fraction']
            item['low_suppressed_fraction']=entry['preservation']['active_frames_suppressed_over_20db_fraction']
        row['systems'][system]=item
    rows.append(row)
assert verified==60
path=f.ROOT/'results/fullband_robustness_audit.json'
path.write_text(json.dumps({'verified_waveforms':verified,'metric_errors':0,'cases':rows},indent=2)+'\n')
for row in rows:
    print(row['case_id'],'original dBFS',round(row['original_clean_rms_dbfs'],2),
          {s:{k:round(v,5) if isinstance(v,float) else v for k,v in d.items() if k!='suppressed_frame_start_seconds'} for s,d in row['systems'].items()},flush=True)
print('Verified',verified,'scored waveforms; no metric errors')
