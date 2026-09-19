"""Separate FC/grouped-FC/1x1-CNN/other-CNN precision ablations."""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from full_probe import FullModel
from probe import FP,ptr,session,initial_state,spectra,summarize,whole_parity,cpu_name


class ExtendedModel(FullModel):
    def __init__(self, reference, tier=0, precision=0, mask=0,
                 build='build/extended', weights='models/extended/weights.f32'):
        path=Path(build)/'libdpdf_full.so'; weights=Path(weights)
        super().__init__(path,weights,reference,tier)
        self.close()
        self.lib.dpdf_model_create_config.argtypes=[FP,ct.c_size_t,ct.c_int,ct.c_int,ct.c_uint]
        self.lib.dpdf_model_create_config.restype=ct.c_void_p
        self.lib.dpdf_model_owned_bytes.argtypes=[ct.c_void_p]
        self.lib.dpdf_model_owned_bytes.restype=ct.c_size_t
        w=np.fromfile(weights,dtype='<f4')
        self.handle=self.lib.dpdf_model_create_config(ptr(w),w.size,tier,precision,mask)
        if not self.handle: raise RuntimeError(f'Create failed: tier={tier}, precision={precision}, mask={mask}')
        self.owned_bytes=int(self.lib.dpdf_model_owned_bytes(self.handle))


CONFIGS={'compact_fp32':(0,0,0),'dprnn_fp16':(3,0,0),'dprnn_int8':(4,0,0)}
for precision in (16,8):
    tier=3 if precision==16 else 4
    for name,mask in [('fc',1),('grouped_fc',2),('cnn_1x1',4),('cnn_other',8),('fc_and_1x1',7),('all',15)]:
        CONFIGS[f'{name}_{precision}']=(tier,precision,mask)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke',action='store_true');p.add_argument('--names',nargs='+')
    a=p.parse_args();frames=spectra(200 if a.smoke else 1100)
    ref=session('models/dpdfnet8_48khz_hr.onnx')
    result={'cpu':cpu_name(),'configs':CONFIGS,'results':{},'precision_notes':'FP16 weight storage/FP32 arithmetic; INT8 dynamic W8A8. State/norm/filter stay FP32.'}
    names=a.names or list(CONFIGS)
    for name in names:
        candidate=ExtendedModel(ref,*CONFIGS[name])
        if name=='compact_fp32': result['parity']=whole_parity(ref,candidate,frames,'synthetic')
        runs=[]
        for repeat in range(1 if a.smoke else 3):
            state=initial_state(ref);times=[]
            for i,x in enumerate(frames):
                begin=time.perf_counter_ns();_,state=candidate.run(None,{'spec':x,'state_in':state})
                elapsed=(time.perf_counter_ns()-begin)/1e6
                assert np.isfinite(state).all()
                if i>=100:times.append(elapsed)
            runs.append(summarize(times))
        result['results'][name]={'owned_bytes':candidate.owned_bytes,'timings':runs}
        print(name,candidate.owned_bytes,runs,flush=True)
        candidate.close()
    path=Path('results/extended_smoke.json' if a.smoke else 'results/extended_ablations.json')
    result['library_sha256']=hashlib.sha256(Path('build/extended/libdpdf_full.so').read_bytes()).hexdigest()
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
