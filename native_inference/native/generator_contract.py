"""Compile generated transposes and a view-lifetime arena case against oracles."""
import ctypes as ct
import itertools
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from generate_model import transpose_code
from arena_planner import reuse_arena


def main():
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        code='#include <string.h>\n#include <stddef.h>\n'
        cases=[]
        for shape in [(1,), (1,3,4), (2,3,4), (1,2,3,1), (2,2,3,4)]:
            for perm in itertools.permutations(range(len(shape))):
                out_shape=tuple(shape[i] for i in perm)
                name=f'test{len(cases)}'
                code+=f'void {name}(const float *x,float *y) {{'+transpose_code(shape,out_shape,perm,'x','y')+'}\n'
                cases.append((name,shape,perm))
        # The second slice reads root 0 late; its storage must survive temporary
        # root 32. Roots 16 and 48 are dead at different points and can be reused.
        toy='''#include <stddef.h>
typedef struct { float *arena; } model;
size_t dpdf_model_arena_bytes(void) { return 80*sizeof(float); }
int dpdf_model_process(model *m,float *out) {
/* 0: Fill A */
for (int i=0;i<16;++i) (m->arena+0)[i]=(float)i;
/* 1: View B */
for (int i=0;i<8;++i) (m->arena+16)[i]=((m->arena+0)+8)[i]*2;
/* 2: Fill C */
for (int i=0;i<16;++i) (m->arena+32)[i]=100+(float)i;
/* 3: Add D */
for (int i=0;i<8;++i) (m->arena+48)[i]=(m->arena+32)[i]+((m->arena+0)+4)[i]+(m->arena+16)[i];
/* 4: Copy E */
for (int i=0;i<8;++i) (m->arena+64)[i]=(m->arena+48)[i];
for (int i=0;i<8;++i) out[i]=(m->arena+64)[i];
return 0;
}
'''
        optimized,_,count,_=reuse_arena(toy,[{'offset':i,'shape':[16]} for i in range(0,80,16)],80)
        assert count<80
        (folder/'transposes.c').write_text(code)
        (folder/'arena.c').write_text(optimized)
        for name in ('transposes','arena'):
            subprocess.run(['cc','-shared','-fPIC','-O2','-Wall','-Wextra','-Werror',str(folder/(name+'.c')),'-o',str(folder/(name+'.so'))],check=True)
        lib=ct.CDLL(str(folder/'transposes.so'))
        ptr=ct.POINTER(ct.c_float)
        for name,shape,perm in cases:
            x=np.arange(np.prod(shape),dtype=np.float32).reshape(shape)
            y=np.empty_like(x).ravel()
            fn=getattr(lib,name);fn.argtypes=[ptr,ptr]
            fn(x.ctypes.data_as(ptr),y.ctypes.data_as(ptr))
            assert y.tobytes()==x.transpose(perm).copy().tobytes(),(shape,perm)
        lib=ct.CDLL(str(folder/'arena.so'))
        class Model(ct.Structure): _fields_=[('arena',ptr)]
        arena=np.full(count,np.nan,dtype=np.float32);out=np.zeros(8,dtype=np.float32)
        model=Model(arena.ctypes.data_as(ptr))
        lib.dpdf_model_process.argtypes=[ct.POINTER(Model),ptr]
        lib.dpdf_model_process(ct.byref(model),out.ctypes.data_as(ptr))
        expected=np.arange(8,dtype=np.float32)*4+120
        assert np.array_equal(out,expected),(out,expected)
        print(f'{len(cases)} compiled transpose permutations and overlapping-view lifetime case passed')


if __name__=='__main__': main()
