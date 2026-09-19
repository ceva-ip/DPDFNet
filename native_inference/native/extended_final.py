"""Rotating continuous/paced comparison against preserved previous C libraries."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import numpy as np
from extended_probe import ExtendedModel,CONFIGS
from full_probe import FullModel,timings
from probe import session,spectra,cpu_name


def main():
    output=Path('results/extended_final.json')
    if '--memory-only' in sys.argv:
        data=json.loads(output.read_text())
        measure_memory(data)
        output.write_text(json.dumps(data,indent=2)+'\n')
        return
    ref=session('models/dpdfnet8_48khz_hr.onnx')
    sessions={'previous_fp32':FullModel(Path('build/full/libdpdf_full.so'),Path('models/full_c/weights.f32'),ref,0),
              'previous_fp16':FullModel(Path('build/full/libdpdf_full.so'),Path('models/full_c/weights.f32'),ref,3),
              'previous_int8':FullModel(Path('build/full/libdpdf_full.so'),Path('models/full_c/weights.f32'),ref,4)}
    for name in ('fc_16','fc_and_1x1_16','fc_8','fc_and_1x1_8'):
        sessions[name]=ExtendedModel(ref,*CONFIGS[name])
    data={'cpu':cpu_name(),'config':{'timed_frames':1000,'warmup':100,'repeats':3,'threads':1}}
    data['artifact_sha256']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
        ('build/full/libdpdf_full.so','build/extended/libdpdf_full.so','models/extended/weights.f32','models/extended/generated_model.c')}
    frames=spectra(1100)
    data['continuous']=timings(sessions,frames,3,False)
    data['paced']=timings(sessions,frames,3,True)
    for obj in sessions.values():obj.close()
    measure_memory(data)
    output.write_text(json.dumps(data,indent=2)+'\n')


def measure_memory(data):
    data['memory']={}
    data['memory_method']='Fresh exec; /proc/self/status VmRSS and VmHWM (not inherited getrusage high-water mark)'
    for name in data['continuous']:
        old=name.startswith('previous_')
        tier={'previous_fp32':0,'previous_fp16':3,'previous_int8':4}.get(name)
        tier,precision,mask=(tier,0,0) if old else CONFIGS[name]
        args=['build/memory_probe',str(Path('build/full' if old else 'build/extended')/'libdpdf_full.so'),
              str(Path('models/full_c' if old else 'models/extended')/'weights.f32'),str(tier),str(precision),str(mask)]
        data['memory'][name]=[json.loads(subprocess.check_output(args,text=True)) for _ in range(3)]


if __name__=='__main__':main()
