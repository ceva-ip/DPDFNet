"""Validate all native DPRNN blocks and measure block + whole-model performance.

Thresholds are numerical engineering gates, not perceptual-quality claims.
Reference and candidate carry independent temporal state across every frame.
"""
import argparse
import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import cpu_name, initial_state, spectra


FP = ct.POINTER(ct.c_float)


def ptr(x):
    assert x.dtype == np.float32 and x.flags.c_contiguous
    return x.ctypes.data_as(FP)


def session(path, custom=None):
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = opt.inter_op_num_threads = 1
    opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if custom:
        opt.register_custom_ops_library(str(Path(custom).resolve()))
    return ort.InferenceSession(str(path), opt, providers=['CPUExecutionProvider'])


def library(path):
    lib = ct.CDLL(str(path.resolve()))
    lib.dpdf_create.argtypes = [ct.c_int, FP, ct.c_size_t, ct.c_float, ct.c_float, ct.c_int]
    lib.dpdf_create.restype = ct.c_void_p
    lib.dpdf_destroy.argtypes = [ct.c_void_p]
    lib.dpdf_destroy.restype = None
    lib.dpdf_tier.argtypes = [ct.c_void_p]
    lib.dpdf_tier.restype = ct.c_char_p
    lib.dpdf_process.argtypes = [ct.c_void_p, FP, FP, FP, FP]
    lib.dpdf_process.restype = ct.c_int
    lib.dpdf_has_avx2.restype = ct.c_int
    lib.dpdf_test_gates.argtypes = [ct.c_int, FP, FP, FP, ct.c_size_t]
    lib.dpdf_test_gates.restype = ct.c_int
    return lib


class Block:
    def __init__(self, lib, folder, desc, tier):
        self.lib, self.f = lib, desc['frequency']
        w = np.fromfile(folder / (desc['name'] + '.f32'), dtype='<f4')
        self.handle = lib.dpdf_create(self.f, ptr(w), w.size, *desc['eps'], tier)
        if not self.handle:
            raise RuntimeError('Native block creation failed')
        self.tier = lib.dpdf_tier(self.handle).decode()
        self.state = np.zeros(self.f * 64, dtype=np.float32)
        self.output = np.empty((1, 64, 1, self.f), dtype=np.float32)
        self._s, self._y = ptr(self.state), ptr(self.output)

    def run(self, x):
        rc = self.lib.dpdf_process(self.handle, ptr(x), self._s, self._y, self._s)
        if rc:
            raise RuntimeError(f'Native block returned {rc}')
        return self.output, self.state

    def reset(self):
        self.state.fill(0)

    def close(self):
        if self.handle:
            self.lib.dpdf_destroy(self.handle)
            self.handle = None


def error_stats():
    return {'max_abs': 0.0, 'error_squared': 0.0, 'reference_squared': 0.0, 'count': 0}


def compare(stats, a, b, atol, rtol):
    assert np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite output'
    diff = a.astype(np.float64) - b
    stats['max_abs'] = max(stats['max_abs'], float(np.abs(diff).max()))
    stats['error_squared'] += float(np.sum(diff*diff))
    stats['reference_squared'] += float(np.sum(a.astype(np.float64)**2))
    stats['count'] += a.size
    np.testing.assert_allclose(b, a, atol=atol, rtol=rtol)


def finish(stats):
    stats['relative_rms'] = float(np.sqrt(stats['error_squared'] / max(stats['reference_squared'], 1e-30)))
    stats['rmse'] = float(np.sqrt(stats['error_squared'] / stats['count']))
    return stats


def check_activations(lib):
    x = np.concatenate([np.linspace(-100,100,200001), np.linspace(-0.01,0.01,20001)]).astype(np.float32)
    y, z = np.empty_like(x), np.empty_like(x)
    assert lib.dpdf_test_gates(0, ptr(x), ptr(y), ptr(z), x.size) == 0
    reference_s = (1/(1+np.exp(-x.astype(np.float64)))).astype(np.float32)
    reference_t = np.tanh(x.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(y, reference_s, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(z, reference_t, atol=2e-7, rtol=2e-7)
    return {'samples': x.size, 'sigmoid_max_abs': float(np.max(np.abs(y-reference_s))),
            'tanh_max_abs': float(np.max(np.abs(z-reference_t)))}


def check_concurrent(lib, folder, desc):
    block=Block(lib,folder,desc,0)
    inputs=np.random.default_rng(99).standard_normal((32,1,64,1,desc['frequency'])).astype(np.float32)
    def run(_):
        state=np.zeros(desc['frequency']*64,dtype=np.float32)
        output=np.empty_like(inputs[0])
        for frame in inputs:
            assert lib.dpdf_process(block.handle,ptr(frame),ptr(state),ptr(output),ptr(state))==0
        return output,state
    expected=run(0)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(run,range(8)):
            for a,b in zip(expected,result): np.testing.assert_array_equal(a,b)
    block.close()
    return {'streams':8,'threads':4,'frames_per_stream':32,'shared_immutable_context':True,'bit_identical':True}


def check_blocks(lib, folder, manifest, frames):
    results = []
    for d in manifest['blocks']:
        oracle = session(folder / (d['name'] + '.onnx'))
        rng = np.random.default_rng(713)
        inputs = rng.standard_normal((frames,1,64,1,d['frequency'])).astype(np.float32)
        # Scales plus silence transitions exercise gates and normalization.
        inputs[:frames//4] *= 0.01
        inputs[frames//2:3*frames//4] *= 5
        inputs[-8:] = 0
        tiers = [1,2] if lib.dpdf_has_avx2() else [1]
        for tier in tiers:
            block = Block(lib, folder, d, tier)
            state = np.zeros(d['frequency']*64, dtype=np.float32)
            errors = [error_stats(), error_stats()]
            for frame in inputs:
                ref_y, state = oracle.run(None, {d['input_name']: frame, d['state_input_name']: state})
                y, s = block.run(frame)
                compare(errors[0], ref_y, y, 3e-4, 3e-4)
                compare(errors[1], state, s, 3e-4, 3e-4)
            # Reset determinism; two instances must not interfere.
            block.reset()
            expected = tuple(v.copy() for v in block.run(inputs[0]))
            other = Block(lib, folder, d, tier)
            for frame in inputs[:3]:
                other.run(frame)
            block.reset()
            actual = block.run(inputs[0])
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a, b)
            other.close()
            results.append({'block': d['name'], 'tier': block.tier, 'frames': frames,
                            'output': finish(errors[0]), 'state': finish(errors[1])})
            block.close()
        print(f'Validated {d["name"]}: scalar and available SIMD tiers', flush=True)
    return results


def summarize(times):
    a = np.asarray(times)
    return {'mean_ms': float(a.mean()), 'p50_ms': float(np.percentile(a,50)),
            'p95_ms': float(np.percentile(a,95)), 'p99_ms': float(np.percentile(a,99)),
            'max_ms': float(a.max()), 'over_10ms': int((a>10).sum())}


def block_timings(lib, folder, manifest, frames, repeats):
    results = {}
    for d in [manifest['blocks'][0], manifest['blocks'][8]]:
        oracle = session(folder / (d['name']+'.onnx'))
        block = Block(lib, folder, d, 0)
        x = np.random.default_rng(51).standard_normal((1,64,1,d['frequency'])).astype(np.float32)
        state = np.zeros(d['frequency']*64, dtype=np.float32)
        runs = {'onnx': [], 'native': []}
        for repeat in range(repeats):
            for mode in (['onnx','native'] if repeat % 2 == 0 else ['native','onnx']):
                state.fill(0); block.reset()
                times = []
                for i in range(frames+100):
                    t = time.perf_counter_ns()
                    if mode == 'native':
                        block.run(x)
                    else:
                        _, state = oracle.run(None, {d['input_name']:x,d['state_input_name']:state})
                    dt = (time.perf_counter_ns()-t)/1e6
                    if i>=100:
                        times.append(dt)
                runs[mode].append(summarize(times))
        results[d['name']] = runs
        block.close()
    return results


def audio_spectra(path):
    import soundfile as sf
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    if rate != 48000 or audio.shape[1] != 1:
        raise ValueError('Audio fixture must be 48 kHz mono; no implicit resampling')
    audio = audio[:,0]
    # Include the full fixture plus drain hops, on the exact causal FFT grid.
    count = (audio.size+479)//480 + 6
    padded = np.pad(audio, (480, count*480-audio.size))
    n = np.arange(960, dtype=np.float64)
    window = np.sin(0.5*np.pi*np.sin(np.pi*(n+0.5)/960)**2).astype(np.float32)
    result = np.empty((count,1,1,481,2), dtype=np.float32)
    for i in range(count):
        z = np.fft.rfft(padded[i*480:i*480+960]*window)
        result[i,0,0,:,0], result[i,0,0,:,1] = z.real, z.imag
    return result


def synthesize(spectra_out):
    n=np.arange(960,dtype=np.float64)
    window=np.sin(0.5*np.pi*np.sin(np.pi*(n+0.5)/960)**2).astype(np.float32)
    ola=np.zeros(960,dtype=np.float32)
    out=[]
    for frame in spectra_out:
        z=frame.reshape(481,2)
        time_frame=np.fft.irfft(z[:,0]+1j*z[:,1],n=960).astype(np.float32)*window
        ola[:480]=ola[480:]; ola[480:]=0; ola+=time_frame
        out.append(ola[:480].copy())
    return np.concatenate(out)


def whole_parity(reference, candidate, frames, label):
    left, right = initial_state(reference), initial_state(candidate)
    stats = [error_stats(),error_stats()]
    outputs = [[],[]]
    for x in frames:
        a,left = reference.run(None, {'spec':x, 'state_in':left})
        b,right = candidate.run(None, {'spec':x, 'state_in':right})
        compare(stats[0],a,b,2e-3,2e-3)
        compare(stats[1],left,right,2e-3,2e-3)
        outputs[0].append(a); outputs[1].append(b)
    pcm = [synthesize(o) for o in outputs]
    error = np.sum((pcm[0].astype(np.float64)-pcm[1])**2)
    energy = np.sum(pcm[0].astype(np.float64)**2)
    snr = float(10*np.log10(max(energy,1e-30)/max(error,1e-30)))
    if energy > 1e-8:
        assert snr > 70, f'Waveform parity SNR below 70 dB: {snr}'
    return {'fixture':label,'frames':len(frames),'spectrum':finish(stats[0]),'state':finish(stats[1]),
            'waveform_snr_db':snr, 'waveform_max_abs':float(np.max(np.abs(pcm[0]-pcm[1])))}


def whole_timings(reference, candidate, frames, repeats, paced=False):
    sessions = {'onnx':reference,'native_hybrid':candidate}
    runs = {k:[] for k in sessions}
    for repeat in range(repeats):
        order = list(sessions) if repeat%2==0 else list(reversed(sessions))
        for name in order:
            sess=sessions[name]; state=initial_state(sess); times=[]; late=[]
            start=time.perf_counter()
            for i,x in enumerate(frames):
                target=start+i*0.01
                if paced:
                    remaining=target-time.perf_counter()
                    if remaining>0: time.sleep(remaining)
                t=time.perf_counter_ns()
                _,state=sess.run(None,{'spec':x,'state_in':state})
                dt=(time.perf_counter_ns()-t)/1e6
                if i>=100:
                    times.append(dt)
                    if paced: late.append(max(0,t/1e9-target)*1000)
            result=summarize(times)
            result['rtf_mean']=result['mean_ms']/10
            if paced: result['start_lateness_p99_ms']=float(np.percentile(late,99))
            runs[name].append(result)
            print(f'{name}, paced={paced}, repeat={repeat}: {result["mean_ms"]:.3f} ms',flush=True)
    return runs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=Path('models/dpdfnet8_48khz_hr.onnx'))
    p.add_argument('--blocks',type=Path,default=Path('models/native_blocks'))
    p.add_argument('--build',type=Path,default=Path('build'))
    p.add_argument('--output',type=Path,default=Path('results/native_prototype.json'))
    p.add_argument('--audio',type=Path,action='append',help='Repeat to validate multiple 48 kHz mono fixtures')
    p.add_argument('--frames',type=int,default=1000)
    p.add_argument('--block-frames',type=int,default=128)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--paced',action='store_true')
    p.add_argument('--check-only',action='store_true',help='Validate without timing measurements')
    a=p.parse_args()
    if a.frames<=0 or a.block_frames<32 or a.repeats<=0: p.error('Invalid frame/repeat counts')
    manifest=json.loads((a.blocks/'manifest.json').read_text())
    assert hashlib.sha256(a.model.read_bytes()).hexdigest()==manifest['source_sha256']
    lib=library(a.build/'libdpdf_dprnn.so')
    result={'environment':{'platform':platform.platform(),'cpu':cpu_name(),'ort':ort.__version__,
                          'python':platform.python_version(),'compiler':subprocess.check_output(['cc','--version'],text=True).splitlines()[0]},
            'source_sha256':manifest['source_sha256'], 'avx2_available':bool(lib.dpdf_has_avx2()),
            'config':{'frames':a.frames,'block_frames':a.block_frames,'repeats':a.repeats,'warmup':100,
                      'threads':1,'precision':'FP32; AVX2 FMA and polynomial activations, no quantization'},
            'activation_test':check_activations(lib)}
    result['block_parity']=check_blocks(lib,a.blocks,manifest,a.block_frames)
    result['concurrency_test']=check_concurrent(lib,a.blocks,manifest['blocks'][0])
    if not a.check_only:
        result['block_timings']=block_timings(lib,a.blocks,manifest,a.frames,a.repeats)
    ref=session(a.model); candidate=session(a.blocks/'hybrid.onnx',a.build/'libdpdf_ort.so')
    frames=spectra(a.frames+100)
    result['whole_parity']=[whole_parity(ref,candidate,frames,'synthetic')]
    for audio in a.audio or []:
        real=audio_spectra(audio)
        parity=whole_parity(ref,candidate,real,audio.name)
        parity['audio_sha256']=hashlib.sha256(audio.read_bytes()).hexdigest()
        result['whole_parity'].append(parity)
        print(f'Validated {audio.name}: waveform parity {parity["waveform_snr_db"]:.1f} dB',flush=True)
    if not a.check_only:
        result['whole_timings']=whole_timings(ref,candidate,frames,a.repeats)
        if a.paced:
            result['paced_timings']=whole_timings(ref,candidate,frames,a.repeats,True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'All numerical gates passed; results: {a.output}',flush=True)


if __name__=='__main__':
    main()
