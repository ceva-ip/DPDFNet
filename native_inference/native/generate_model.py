"""Compile the audited static ONNX graph into straight-line C and FP32 weights.

This is NOT a general ONNX runtime. Unsupported operators/shapes/attributes
fail at export. Source is hash-pinned and regenerated from the audited exporter.
Integer shape/index tensors are compile-time only. Views alias existing storage;
all other tensors have fixed, disjoint arena slots for straightforward auditing.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from export_blocks import export, attributes


def strides(shape):
    return [math.prod(shape[i+1:]) for i in range(len(shape))]


def coord(shape, axis, index='i'):
    if shape[axis] == 1:
        return '0'
    return f'(({index}/{strides(shape)[axis]})%{shape[axis]})'


def broadcast_index(source, dest, index='i'):
    if source == dest:
        return index
    padded = [1]*(len(dest)-len(source)) + source
    if len(padded) != len(dest) or any(a != b and a != 1 for a,b in zip(padded,dest)):
        raise ValueError(f'Unsupported broadcast {source} -> {dest}')
    terms = [f'{coord(dest,j,index)}*{s}' for j,(d,s) in enumerate(zip(padded,strides(padded))) if d != 1]
    return '+'.join(terms) or '0'


def generate(source, folder):
    export(source, folder/'oracles')
    oracle_manifest=json.loads((folder/'oracles'/'manifest.json').read_text())
    model=onnx.shape_inference.infer_shapes(onnx.load(str(folder/'oracles/hybrid.onnx')))
    onnx.checker.check_model(model)
    const={x.name:numpy_helper.to_array(x) for x in model.graph.initializer}
    shape={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim]
           for v in list(model.graph.value_info)+list(model.graph.input)+list(model.graph.output)}
    shape.update({x.name:list(x.dims) for x in model.graph.initializer})
    if any(any(d<=0 for d in s) for s in shape.values()):
        raise ValueError('All dimensions must be static and positive')
    consumers=defaultdict(list)
    for consumer in model.graph.node:
        for index,name in enumerate(consumer.input):
            consumers[name].append((consumer,index))
    chained_dprnn_outputs=set()
    for producer in model.graph.node:
        if producer.op_type!='DpdfDprnn': continue
        uses=consumers[producer.output[0]]
        if len(uses)==1 and uses[0][0].op_type=='DpdfDprnn' and uses[0][1]==0:
            chained_dprnn_outputs.add(producer.output[0])
    ptr={'spec':'spec','state_in':'state_in'}
    chunks=[]; weight_count=0; arena_count=0; calls=[]; blocks=[]; tensors=[]; weight_layout=[]
    def weight(array):
        nonlocal weight_count
        padding=(-weight_count)%8
        if padding:
            chunks.append(np.zeros(padding,dtype='<f4')); weight_count+=padding
        array=np.ascontiguousarray(array,dtype='<f4').ravel()
        if not np.isfinite(array).all(): raise ValueError('Nonfinite weights')
        expr=f'(m->weights+{weight_count})'
        weight_layout.append({'offset':int(weight_count),'count':int(array.size)})
        weight_count+=array.size; chunks.append(array)
        return expr
    needed=set()
    for node in model.graph.node:
        a=attributes(node)
        repack=node.op_type=='Gemm' or (node.op_type=='Conv' and a['kernel_shape']==[1,1] and a['strides']==[1,1] and a['pads']==[0,0,0,0] and a.get('group',1)==1)
        needed.update(x for j,x in enumerate(node.input) if not (repack and j==1))
    for name,value in const.items():
        if value.dtype==np.float32 and name in needed: ptr[name]=weight(value)
    def allocate(name):
        nonlocal arena_count
        arena_count=(arena_count+15)//16*16
        ptr[name]=f'(m->arena+{arena_count})'
        count=math.prod(shape[name]); arena_count+=count
        tensors.append({'name':name,'offset':arena_count-count,'shape':shape[name]})
        return ptr[name]
    def element(name, output, index='i'):
        return f'{ptr[name]}[{broadcast_index(shape[name],shape[output],index)}]'
    for number,node in enumerate(model.graph.node):
        op=node.op_type; a=attributes(node); ins=list(node.input); outs=list(node.output)
        calls.append(f'/* {number}: {op} {node.name} */')
        if op in ('Reshape','Unsqueeze','Squeeze','Flatten'):
            assert math.prod(shape[ins[0]])==math.prod(shape[outs[0]])
            ptr[outs[0]]=ptr[ins[0]]
            continue
        dest=[allocate(x) for x in outs]; y=dest[0]; out=outs[0]; os=shape[out]; size=math.prod(os)
        if op in ('Add','Sub','Mul','Div','Pow'):
            lhs,rhs=element(ins[0],out),element(ins[1],out)
            if op=='Pow':
                if ins[1] not in const or float(const[ins[1]])!=2: raise ValueError('Expected squared magnitude')
                expr=f'{lhs}*{lhs}'
            else: expr=f'{lhs}{dict(Add="+",Sub="-",Mul="*",Div="/")[op]}{rhs}'
            calls.append(f'for (int i=0;i<{size};++i) {y}[i]={expr};')
        elif op in ('Sqrt','Log','Relu','Sigmoid','Tanh'):
            x=element(ins[0],out)
            expr={'Sqrt':f'sqrtf({x})','Log':f'logf({x})','Relu':f'({x}>0?{x}:0.0f)',
                  'Sigmoid':f'1.0f/(1.0f+expf(-{x}))','Tanh':f'tanhf({x})'}[op]
            calls.append(f'for (int i=0;i<{size};++i) {y}[i]={expr};')
        elif op=='DpdfDprnn':
            idx=len(blocks); w=weight(np.asarray(a['weights'],dtype=np.float32))
            blocks.append((w,a['freq'],a['intra_epsilon'],a['inter_epsilon']))
            layout=(1 if ins[0] in chained_dprnn_outputs else 0) | (2 if outs[0] in chained_dprnn_outputs else 0)
            calls.append(f'if (dpdf_process_layout(m->blocks[{idx}],{ptr[ins[0]]},{ptr[ins[1]]},{y},{dest[1]},{layout})) return -1;')
        elif op=='Transpose':
            perm=a['perm']; ss=shape[ins[0]]; st=strides(ss)
            assert sorted(perm)==list(range(len(ss))) and os==[ss[j] for j in perm]
            index='+'.join(f'{coord(os,j)}*{st[k]}' for j,k in enumerate(perm) if ss[k]>1) or '0'
            calls.append(f'for (int i=0;i<{size};++i) {y}[i]={ptr[ins[0]]}[{index}];')
        elif op=='Slice':
            ss=shape[ins[0]]; rank=len(ss); start=[0]*rank; step=[1]*rank
            axes=const[ins[3]].ravel() if len(ins)>3 else range(len(const[ins[1]].ravel()))
            steps=const[ins[4]].ravel() if len(ins)>4 else [1]*len(axes)
            slices=[slice(None)]*rank
            for axis,begin,end,inc in zip(axes,const[ins[1]].ravel(),const[ins[2]].ravel(),steps):
                axis=int(axis)%rank; slices[axis]=slice(int(begin),int(end),int(inc))
            dims=[]
            for j,s in enumerate(slices):
                start[j],end,step[j]=s.indices(ss[j]); dims.append(len(range(start[j],end,step[j])))
            assert dims==os
            st=strides(ss)
            index='+'.join(f'({start[j]}+{coord(os,j)}*{step[j]})*{st[j]}' for j in range(rank))
            # Alias a slice only when its flattened memory span is contiguous.
            offsets=[start[j]*st[j] for j in range(rank)]
            contiguous=all(d<=1 or inc*stride==outstride for d,inc,stride,outstride in zip(os,step,st,strides(os)))
            if contiguous:
                ptr[out]=f'({ptr[ins[0]]}+{sum(offsets)})'
            else:
                calls.append(f'for (int i=0;i<{size};++i) {y}[i]={ptr[ins[0]]}[{index}];')
        elif op=='Gather':
            ss=shape[ins[0]]; axis=a['axis']%len(ss); indices=const[ins[1]]
            if indices.ndim!=0: raise ValueError('Only scalar Gather is supported')
            index=int(indices)%ss[axis]; before=math.prod(ss[:axis]); after=math.prod(ss[axis+1:])
            assert size==before*after
            calls.append(f'for (int b=0;b<{before};++b) memcpy({y}+b*{after},{ptr[ins[0]]}+(b*{ss[axis]}+{index})*{after},{after}*sizeof(float));')
        elif op in ('Concat','Split'):
            ss=shape[ins[0]]; axis=a['axis']%len(ss)
            if op=='Concat':
                outer=math.prod(os[:axis]); inner=math.prod(os[axis+1:]); offset=0
                for name in ins:
                    count=shape[name][axis]*inner
                    calls.append(f'for (int b=0;b<{outer};++b) memcpy({y}+b*{os[axis]*inner}+{offset},{ptr[name]}+b*{count},{count}*sizeof(float));')
                    offset+=count
                assert offset==os[axis]*inner
            else:
                outer=math.prod(ss[:axis]); inner=math.prod(ss[axis+1:]); offset=0
                for name,d in zip(outs,dest):
                    count=shape[name][axis]*inner
                    calls.append(f'for (int b=0;b<{outer};++b) memcpy({d}+b*{count},{ptr[ins[0]]}+b*{ss[axis]*inner}+{offset},{count}*sizeof(float));')
                    offset+=count
                assert offset==ss[axis]*inner
        elif op=='ReduceSum':
            ss=shape[ins[0]]; axes=[int(v)%len(ss) for v in const[ins[1]].ravel()]
            if len(axes)!=1: raise ValueError('Only single-axis ReduceSum is supported')
            axis=axes[0]; before=math.prod(ss[:axis]); after=math.prod(ss[axis+1:]); red=ss[axis]
            assert size==before*after
            calls.append(f'for (int b=0;b<{before};++b) for (int j=0;j<{after};++j) {{ float sum=0; for (int k=0;k<{red};++k) sum+={ptr[ins[0]]}[(b*{red}+k)*{after}+j]; {y}[b*{after}+j]=sum; }}')
        elif op=='Pad':
            ss=shape[ins[0]]; pads=const[ins[1]].tolist(); rank=len(ss)
            assert a['mode']==b'reflect' and all(v==0 for v in pads[:-1]) and pads[-1]==1
            assert os[:-1]==ss[:-1] and os[-1]==ss[-1]+1
            width=ss[-1]; outer=math.prod(ss[:-1])
            calls.append(f'for (int b=0;b<{outer};++b) {{ memcpy({y}+b*{width+1},{ptr[ins[0]]}+b*{width},{width}*sizeof(float)); {y}[b*{width+1}+{width}]={ptr[ins[0]]}[b*{width}+{width-2}]; }}')
        elif op=='Gemm':
            assert a.get('transB')==1 and a.get('transA',0)==0 and a.get('alpha',1)==a.get('beta',1)==1
            ss=shape[ins[0]]; ws=shape[ins[1]]; assert len(ss)==len(ws)==2 and ws[1]==ss[1]
            bias=ptr[ins[2]]; rows,k=ss; n=ws[0]
            assert rows==1 and n%64==0
            w=weight(const[ins[1]].T.reshape(k,n//64,64).transpose(1,0,2))
            calls.append(f'for (int t=0;t<{n//64};++t) m->affine({ptr[ins[0]]},{w}+t*{k*64},{bias}+t*64,{y}+t*64,1,{k},64);')
        elif op=='MatMul':
            ss=shape[ins[0]]; ws=shape[ins[1]]
            assert len(ss)==4 and len(ws)==3 and ss[0]==1 and ss[1]==ws[0] and ss[-1]==ws[-2]
            groups,rows,k,n=ws[0],ss[-2],ss[-1],ws[-1]
            zero=weight(np.zeros(n,dtype=np.float32))
            fn='m->affine'
            calls.append(f'for (int g=0;g<{groups};++g) {fn}({ptr[ins[0]]}+g*{rows*k},{ptr[ins[1]]}+g*{k*n},{zero},{y}+g*{rows*n},{rows},{k},{n});')
        elif op=='Conv':
            ss=shape[ins[0]]; ws=shape[ins[1]]; assert ss[0]==os[0]==1 and len(ss)==len(os)==len(ws)==4
            assert a['dilations']==[1,1] and a['pads'][:2]==a['pads'][2:]
            bias=ptr[ins[2]] if len(ins)>2 else 'NULL'
            if a['kernel_shape']==[1,1] and a['strides']==[1,1] and a['pads']==[0,0,0,0] and a.get('group',1)==1:
                ci,co,spatial=ss[1],os[1],math.prod(ss[2:])
                packed=weight(const[ins[1]].reshape(co,ci).T)
                if bias=='NULL': bias=weight(np.zeros(co,dtype=np.float32))
                # Fixed scratch slots. Both transposes are outside the MAC loop.
                scratch_in=f'(m->arena+{arena_count})'; arena_count+=ci*spatial
                scratch_out=f'(m->arena+{arena_count})'; arena_count+=co*spatial
                calls.append(f'for (int r=0;r<{spatial};++r) for (int c=0;c<{ci};++c) {scratch_in}[r*{ci}+c]={ptr[ins[0]]}[c*{spatial}+r];')
                calls.append(f'm->affine({scratch_in},{packed},{bias},{scratch_out},{spatial},{ci},{co});')
                calls.append(f'for (int c=0;c<{co};++c) for (int r=0;r<{spatial};++r) {y}[c*{spatial}+r]={scratch_out}[r*{co}+c];')
                continue
            params=[*ss[1:],*os[1:],*a['kernel_shape'],*a['strides'],*a['pads'][:2],a.get('group',1)]
            calls.append(f'dpdf_conv(m->axpy,{ptr[ins[0]]},{ptr[ins[1]]},{bias},{y},'+','.join(map(str,params))+');')
        else:
            raise ValueError(f'Unsupported operator: {op}')
    weights=np.concatenate(chunks).astype('<f4'); weights.tofile(folder/'weights.f32')
    sha=hashlib.sha256(weights.tobytes()).hexdigest()
    block_count=len(blocks)
    if block_count != len(oracle_manifest['blocks']):
        raise ValueError('Generated DPRNN block count does not match oracle manifest')
    creates=[]
    for j,(w,f,e0,e1) in enumerate(blocks):
        creates.append(f'm->blocks[{j}]=tier==DPDF_EXPERIMENTAL_INT8 ? dpdf_create_int8({f},{w},DPDF_WEIGHT_FLOATS,{e0:.9g}f,{e1:.9g}f) : tier==DPDF_EXPERIMENTAL_FP16 ? dpdf_create_fp16({f},{w},DPDF_WEIGHT_FLOATS,{e0:.9g}f,{e1:.9g}f) : dpdf_create({f},{w},DPDF_WEIGHT_FLOATS,{e0:.9g}f,{e1:.9g}f,tier); if (!m->blocks[{j}]) {{ dpdf_model_destroy(m); return NULL; }}')
    code='''/* Generated by generate_model.py from the SHA-pinned model. Do not edit. */
#include "full_model.h"
#include "full_ops.h"
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
struct dpdf_model {
    float *weights, *arena;
    void *weight_allocation;
    dpdf_block *blocks[DPDF_BLOCK_COUNT];
    dpdf_affine_fn affine;
    dpdf_axpy_fn axpy;
};
'''
    code=code.replace('DPDF_BLOCK_COUNT',str(block_count))
    code+=f'const char *dpdf_model_weights_sha256(void) {{ return "{sha}"; }}\n'
    code+=f'size_t dpdf_model_weight_count(void) {{ return {weight_count}; }}\n'
    code+=f'size_t dpdf_model_arena_bytes(void) {{ return {arena_count}*sizeof(float); }}\n'
    meta={p.key:p.value for p in model.metadata_props}
    state_size=math.prod(shape['state_in'])
    spectrum_size=math.prod(shape['spec_e'])
    assert shape['state_out']==shape['state_in'] and int(meta['state_size'])==state_size
    assert shape['spec']==shape['spec_e'] and spectrum_size==962
    code+=f'size_t dpdf_model_state_size(void) {{ return {state_size}; }}\n'
    seeds=[]
    for prefix in ('erb','spec'):
        values=[float(v) for v in meta[f'{prefix}_norm_init'].split(',')]
        assert len(values)==int(meta[f'{prefix}_norm_state_size'])
        seeds.extend(values)
    assert len(seeds)==577
    code+='int dpdf_model_init_state(float *state) {\nif (!state) return -1;\n'
    code+='static const float seeds[]={'+','.join(f'{v:.9e}f' for v in seeds)+'};\n'
    code+=f'memset(state,0,{state_size}*sizeof(float)); memcpy(state,seeds,sizeof(seeds)); return 0;\n}}\n'
    code+='''void dpdf_model_destroy(dpdf_model *m) {
    if (!m) return;
    for (int i=0;i<DPDF_BLOCK_COUNT;++i) dpdf_destroy(m->blocks[i]);
    free(m->weight_allocation); free(m->arena); free(m);
}
dpdf_model *dpdf_model_create(const float *w,size_t count,int tier) {
    if (!w || count!=dpdf_model_weight_count() || tier<0 || tier>DPDF_EXPERIMENTAL_INT8) return NULL;
    if (tier==DPDF_AUTO) tier=dpdf_has_avx2()?DPDF_AVX2:DPDF_SCALAR;
    if (tier==DPDF_AVX2 && !dpdf_has_avx2()) return NULL;
    if (tier==DPDF_EXPERIMENTAL_FP16 && !dpdf_has_fp16()) return NULL;
    if (tier==DPDF_EXPERIMENTAL_INT8 && !dpdf_has_avx2()) return NULL;
    for (size_t i=0;i<count;++i) if (!isfinite(w[i])) return NULL;
    dpdf_model *m=calloc(1,sizeof(*m)); if (!m) return NULL;
    m->weight_allocation=malloc(count*sizeof(float)+31); m->arena=malloc(dpdf_model_arena_bytes());
    if (!m->weight_allocation || !m->arena) { dpdf_model_destroy(m); return NULL; }
    m->weights=(float *)(((uintptr_t)m->weight_allocation+31)&~(uintptr_t)31);
    memcpy(m->weights,w,count*sizeof(float));
    m->affine=dpdf_affine_scalar; m->axpy=dpdf_axpy_scalar;
#ifdef DPDF_X86_DISPATCH
    if (tier>=DPDF_AVX2) { m->affine=dpdf_affine_avx2; m->axpy=dpdf_axpy_avx2; }
#endif
'''
    code=code.replace('DPDF_BLOCK_COUNT',str(block_count))
    code+='\n'.join(creates)+'\nreturn m;\n}\n'
    code+='''int dpdf_model_process(dpdf_model *m,const float *spec,const float *state_in,float *spec_out,float *state_out) {
    if (!m || !spec || !state_in || !spec_out || !state_out) return -1;
'''
    code+='\n'.join(calls)+'\n'
    code+=f'memcpy(spec_out,{ptr["spec_e"]},{spectrum_size}*sizeof(float));\nmemcpy(state_out,{ptr["state_out"]},{state_size}*sizeof(float));\nreturn 0;\n}}\n'
    (folder/'generated_model.c').write_text(code)
    (folder/'manifest.json').write_text(json.dumps({'source_sha256':oracle_manifest['source_sha256'],
        'profile':oracle_manifest['profile'],'state_size':state_size,'spectrum_size':spectrum_size,
        'block_count':block_count,'weights_sha256':sha,
        'weight_floats':int(weight_count),'arena_bytes':int(arena_count*4),'nodes':len(model.graph.node),
        'tensor_layout':tensors,'weight_layout':weight_layout},indent=2)+'\n')
    print(f'Generated complete C spectral graph: {weight_count*4} weight bytes, {arena_count*4} arena bytes')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path); p.add_argument('output',type=Path)
    a=p.parse_args(); generate(a.source,a.output)
