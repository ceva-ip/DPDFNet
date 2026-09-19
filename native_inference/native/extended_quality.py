"""Per-family screen on cafe speech, then seven-mixture evaluation of finalists."""
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import correlate,correlation_lags
from extended_probe import ExtendedModel,CONFIGS
from quality_probe import fixtures,metrics
from probe import session,initial_state,audio_spectra,synthesize


def enhance(model,frames,ref):
    state=initial_state(ref);out=[]
    for x in frames:
        y,state=model.run(None,{'spec':x,'state_in':state})
        assert np.isfinite(y).all() and np.isfinite(state).all()
        out.append(y)
    return synthesize(out)


def main():
    folder=Path('scratch/extended_audio');folder.mkdir(parents=True,exist_ok=True)
    ref=session('models/dpdfnet8_48khz_hr.onnx')
    selected=['fc_16','fc_and_1x1_16','fc_8','fc_and_1x1_8']
    result={'scope':'One speaker; all operator-family configurations on cafe, four finalists on seven mixtures',
            'selected':selected,'fixtures':[]}
    paths=fixtures(folder)
    # Screen broad CNN conversion too, without pretending it is a speed win.
    for path,clean in sorted(paths,key=lambda item:0 if item[0].name=='noisy_cafe_48k.flac' else 1):
        frames=audio_spectra(path);original=enhance(ref,frames,ref)
        names=list(CONFIGS) if path.name=='noisy_cafe_48k.flac' else selected
        item={'fixture':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'candidates':{}}
        lag=None
        if clean is not None:
            xc=correlate(original,clean,method='fft');lags=correlation_lags(len(original),len(clean));mask=(lags>=0)&(lags<=4800)
            lag=int(lags[mask][np.argmax(xc[mask])]);assert 0<lag<4800
            item['alignment_samples']=lag;item['clean_reference_original']=metrics(clean,original[lag:lag+len(clean)])
        for name in names:
            model=ExtendedModel(ref,*CONFIGS[name])
            out=enhance(model,frames,ref);model.close()
            entry={'fidelity':metrics(original,out)}
            if clean is not None:
                entry['clean_reference']=metrics(clean,out[lag:lag+len(clean)])
                entry['delta']={k:v-item['clean_reference_original'][k] for k,v in entry['clean_reference'].items()}
            item['candidates'][name]=entry
            sf.write(folder/f'{path.stem}_{name}.wav',out,48000,subtype='FLOAT')
            print(path.name,name,entry,flush=True)
        result['fixtures'].append(item)
        Path('results/extended_quality.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
