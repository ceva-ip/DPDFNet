"""Independent-stream concurrency for the compact FC/CNN precision contexts."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
from extended_probe import ExtendedModel,CONFIGS
from probe import session,initial_state,spectra


def main():
    ref=session('models/dpdfnet8_48khz_hr.onnx');frames=spectra(64);report={}
    for name in ('compact_fp32','fc_and_1x1_16','fc_and_1x1_8','all_16','all_8'):
        def run(stream):
            m=ExtendedModel(ref,*CONFIGS[name]);state=initial_state(ref);out=[]
            try:
                for x in frames:
                    y,state=m.run(None,{'spec':np.ascontiguousarray(x*(stream+1)/4),'state_in':state});out.append(y)
                return np.stack(out),state
            finally:m.close()
        serial=[run(i) for i in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:parallel=list(pool.map(run,range(4)))
        assert all(np.array_equal(a,b) for left,right in zip(serial,parallel) for a,b in zip(left,right))
        report[name]={'streams':4,'frames':64,'bit_identical':True}
        print(name,'passed',flush=True)
    Path('results/extended_state.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
