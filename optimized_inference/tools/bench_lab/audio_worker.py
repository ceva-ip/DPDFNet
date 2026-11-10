#!/usr/bin/python3
"""Measure the documented NumPy FFT -> ready library -> IFFT host callback."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time,json,array,hashlib,statistics as stats,gc,resource,ctypes as ct
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'examples'))
from dpdfnet import Model
from audio_pipeline import AudioPipeline
libc=ct.CDLL(None,use_errno=True)
libc.personality.argtypes=[ct.c_ulong];libc.personality.restype=ct.c_int

def raw_ns():return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
def feedback(cpu):
    a=Path(f'/sys/devices/system/cpu/cpu{cpu}/acpi_cppc/feedback_ctrs').read_text().split()
    return tuple(int(v.split(':')[1]) for v in a)
def digest(model):
    h=1469598103934665603
    for a in [model.state,model.output]:
        for u in a.view(np.uint32).flat:h=((h^int(u))*1099511628211)&((1<<64)-1)
    return f'{h:016x}'
def perf_command(ctl,ack,s):
    if not ctl:return
    ctl.write(s+'\n');ctl.flush()
    first=ack.read(1)
    while first==b'\x00':first=ack.read(1)
    if first+b''.join(ack.read(1) for _ in range(3))!=b'ack\n':raise RuntimeError('Bad perf acknowledgement')

def main():
    if len(sys.argv)!=9:raise ValueError('library model weights decoded-pcm calls paced:ns 1 raw-output')
    library,model_id,weights,source,calls,mode,batch,output=sys.argv[1:]
    calls=int(calls);model_id=int(model_id)
    if int(batch)!=1 or calls<16 or not mode.startswith('paced:'):raise ValueError('Audio callback requires individual paced samples')
    period=int(mode.split(':')[1]);audio=np.fromfile(source,dtype='<f4')
    if not 1000000<=period<=100000000:raise ValueError('Invalid cadence')
    ctl=ack=None
    if os.getenv('DPDF_PERF_CONTROL'):
        ctl=open(os.environ['DPDF_PERF_CONTROL'],'w');ack=open(os.environ['DPDF_PERF_ACK'],'rb',buffering=0)
    if libc.prctl(29,1,0,0,0):raise OSError('Cannot set timerslack')
    with Model(library,weights,model_id) as model:
        pipeline=AudioPipeline(model,audio)
        times=[0.]*calls;warmup=128
        gc.collect();gc.disable()
        for i in range(warmup):pipeline.process(i)
        cpu=libc.sched_getcpu();ref0,del0=feedback(cpu)
        perf_command(ctl,ack,'enable');before=resource.getrusage(resource.RUSAGE_SELF)
        cpu0=time.thread_time_ns();deadline=time.monotonic_ns();late=0
        for i in range(calls):
            # Absolute deadlines with the same reset-after-lateness policy.
            target=ct.c_long*2;t=target(deadline//1000000000,deadline%1000000000)
            while True:
                rc=libc.clock_nanosleep(1,1,ct.byref(t),None)
                if rc==0:break
                if rc!=4:raise OSError(rc,'clock_nanosleep')
            t0=raw_ns();pipeline.process(warmup+i);t1=raw_ns()
            times[i]=(t1-t0)/1e6
            finish=time.monotonic_ns();late+=finish>deadline+period;deadline+=period
            if finish>deadline:deadline=finish
        cpu1=time.thread_time_ns();after=resource.getrusage(resource.RUSAGE_SELF);perf_command(ctl,ack,'disable')
        ref1,del1=feedback(cpu)
        Path(output).write_bytes(array.array('d',times).tobytes());ordered=sorted(times)
        status=Path('/proc/self/status').read_text();nnp=int(next(x.split()[1] for x in status.splitlines() if x.startswith('NoNewPrivs:')))
        result=dict(cpu=cpu,uid=os.getuid(),no_new_privs=nnp,layout_randomization_disabled=bool(libc.personality(0xffffffff)&0x40000),
            api_address=hex(ct.cast(model.lib.dpdfnet_process,ct.c_void_p).value),state_address=hex(model.state.ctypes.data),
            reference_delta=ref1-ref0,delivered_delta=del1-del0,hardware_feedback_available=ref1>ref0 and del1>del0,
            schema_version=1,period_ns=period,mode=mode,model=model_id,calls=calls,warmup=warmup,observations=calls,batch_size=1,observation_unit='call',
            mean_ms=stats.mean(times),sample_sd_ms=stats.stdev(times),cpu_ms_per_call=(cpu1-cpu0)/1e6/calls,
            p95_ms=ordered[int(calls*.95)],p99_ms=ordered[int(calls*.99)],max_ms=max(times),over_10ms=sum(t>=10 for t in times),late=late,
            voluntary_switches=after.ru_nvcsw-before.ru_nvcsw,involuntary_switches=after.ru_nivcsw-before.ru_nivcsw,
            minor_faults=after.ru_minflt-before.ru_minflt,major_faults=after.ru_majflt-before.ru_majflt,
            checksum=digest(model),pipeline='documented_numpy_fft_nn_ifft',numpy_version=np.__version__)
        print(json.dumps(result))
    if ctl:ctl.close();ack.close()
if __name__=='__main__':main()
