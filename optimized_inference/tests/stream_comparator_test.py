"""Black-box check: catch an intermediate fault in an additional candidate."""
import argparse
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runner',required=True)
    p.add_argument('--baseline',required=True)
    p.add_argument('--fault',required=True)
    p.add_argument('--weights',required=True)
    args=p.parse_args()
    with tempfile.TemporaryDirectory(prefix='dpdf-stream-test-') as folder:
        stream=Path(folder)/'input.f32'
        with stream.open('wb') as f:
            for hop in range(180):
                frame=[]
                for k in range(481):frame.extend([.1*math.sin((hop+1)*(k+1)*.01),0 if k in (0,480) else .1*math.cos((hop+1)*(k+1)*.01)])
                f.write(struct.pack('<962f',*frame))
        command=[args.runner,args.baseline,args.baseline,'2',args.weights,str(stream)]
        good=subprocess.run(command+[args.baseline],capture_output=True,text=True)
        if good.returncode:raise AssertionError(good.stderr)
        env=os.environ.copy();env['DPDF_FAULT_REFERENCE']=args.baseline
        bad=subprocess.run(command+[args.fault],capture_output=True,text=True,env=env)
        if bad.returncode!=1 or 'candidate 2 mismatch at hop 128' not in bad.stderr:
            raise AssertionError((bad.returncode,bad.stdout,bad.stderr))
    print('N-way comparator accepts identical histories and catches the single intermediate bit fault.')

if __name__=='__main__':main()
