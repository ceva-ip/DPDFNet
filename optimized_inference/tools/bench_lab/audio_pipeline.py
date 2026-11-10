"""The documented 48 kHz NumPy host pipeline, shared by QA and callback timing.

This research adapter does not change the runtime ABI or add FFT dependencies
into libdpdfnet. The borrowed returned chunk is valid until the next call.
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'examples'))
def window():
    n=np.arange(960,dtype=np.float64)
    return np.sin(.5*np.pi*np.sin(np.pi*(n+.5)/960)**2).astype(np.float32)

class AudioPipeline:
    def __init__(self,model,audio):
        if audio.dtype!=np.float32 or audio.ndim!=1 or not np.isfinite(audio).all():
            raise ValueError('Expected finite decoded float32 mono PCM at 48 kHz')
        if model.info.sample_rate!=48000 or model.info.hop_samples!=480:
            raise ValueError('Unsupported audio model metadata')
        self.model=model;self.samples=len(audio);self.frames=(len(audio)+479)//480+6
        self.padded=np.pad(audio,(480,self.frames*480-len(audio)))
        self.window=window();self.overlap=np.zeros(960,dtype=np.float32)
        self.last_input=None

    def process(self,index):
        i=index%self.frames
        value=np.fft.rfft(self.padded[i*480:i*480+960]*self.window)
        spectrum=np.empty((481,2),dtype=np.float32)
        spectrum[:,0],spectrum[:,1]=value.real,value.imag
        self.last_input=spectrum
        output=self.model.process(spectrum)
        frame=np.fft.irfft(output[:,0]+1j*output[:,1],n=960).astype(np.float32)*self.window
        self.overlap[:480]=self.overlap[480:];self.overlap[480:]=0;self.overlap+=frame
        return self.overlap[:480]
