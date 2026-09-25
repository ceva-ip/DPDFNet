"""Extend the pinned C graph with per-family precision and compact owned weights.

Transforms only the exact straight-line generator syntax; fails on mismatch.
The input blob is unchanged, but matrices copied into operator contexts are
not retained a second time in the graph's constant array.
"""
import argparse
import json
import re
from pathlib import Path
from generate_model import generate
from generated_api import finalize, validate_prefix


def extend(source, folder, symbol_prefix='dpdf_model'):
    validate_prefix(symbol_prefix)
    generate(source, folder, finalize_output=False)
    path=folder/'generated_model.c'; code=path.read_text()
    manifest=json.loads((folder/'manifest.json').read_text())
    block_count=manifest['block_count']
    prefix,body=code.split('int dpdf_model_process(',1)
    parts=re.split(r'(/\* \d+: \w+ [^\n]*\*/)',body)
    dense=[]; conv=[]; families=[]
    def weight_arg(s):
        assert 'm->weights' in s or s=='NULL', s
        return s.replace('m->weights','w')
    def create_dense(w,bias,k,n,tiled,family,groups=1,stride=0):
        base=len(dense)
        for g in range(groups):
            index=len(dense)
            ws=weight_arg(w)+(f'+{g*stride}' if g else '')
            dense.append(f'm->dense[{index}]=dpdf_dense_create({ws},{weight_arg(bias)},{k},{n},(mask&{family})?precision:0,{tiled}); if (!m->dense[{index}]) {{ dpdf_model_destroy(m);return NULL; }}')
            families.append({'kind':'dense','index':index,'family':family,'k':k,'n':n})
        return base
    for i in range(1,len(parts),2):
        comment,section=parts[i:i+2]
        if ': Gemm ' in comment:
            line=re.search(r'for \(int t=0;t<(\d+);\+\+t\) m->affine\(([^\n]+)\);',section)
            assert line, comment
            args=line[2].split(','); assert len(args)==7 and args[-1]=='64' and args[-3]=='1'
            x,w,b,y,_,k,_=args;k=int(k);n=int(line[1])*64
            w=w.split('+t*')[0]; b=b.split('+t*')[0];y=y.split('+t*')[0]
            idx=create_dense(w,b,k,n,64,1)
            section=section.replace(line[0],f'dpdf_dense_run(m->dense[{idx}],{x},{y},1);')
        elif ': MatMul ' in comment:
            line=re.search(r'for \(int g=0;g<(\d+);\+\+g\) m->affine\(([^\n]+)\);',section)
            assert line, comment
            x,w,b,y,rows,k,n=line[2].split(',');k=int(k);n=int(n)
            assert w.endswith(f'+g*{k*n}')
            idx=create_dense(w.rsplit('+g*',1)[0],b,k,n,0,2,int(line[1]),k*n)
            section=section.replace(line[0],f'for (int g=0;g<{line[1]};++g) dpdf_dense_run(m->dense[{idx}+g],{x},{y},{rows});')
        elif ': Conv ' in comment:
            if 'm->affine(' in section:
                line=re.search(r'm->affine\(([^\n]+)\);',section)
                x,w,b,y,rows,k,n=line[1].split(',')
                idx=create_dense(w,b,int(k),int(n),0,4)
                section=section.replace(line[0],f'dpdf_dense_run(m->dense[{idx}],{x},{y},{rows});')
            else:
                line=re.search(r'dpdf_conv\(([^\n]+)\);',section);assert line,comment
                axpy,x,w,b,y,*params=line[1].split(',');assert axpy=='m->axpy' and len(params)==13
                idx=len(conv)
                conv.append(f'm->conv[{idx}]=dpdf_convop_create({weight_arg(w)},{weight_arg(b)},'+','.join(params)+f',(mask&8)?precision:0); if (!m->conv[{idx}]) {{ dpdf_model_destroy(m);return NULL; }}')
                families.append({'kind':'conv','index':idx,'family':8,'parameters':list(map(int,params))})
                section=section.replace(line[0],f'dpdf_convop_run(m->conv[{idx}],{x},{y});')
        parts[i+1]=section
    body=''.join(parts)
    # Keep only constants still directly read by graph execution. Contexts own
    # their matrix/bias copies; create reads them from the caller's input blob.
    used={int(v) for v in re.findall(r'm->weights\+(\d+)',body)}
    counts={x['offset']:x['count'] for x in manifest['weight_layout']}
    assert used <= counts.keys()
    compact=0; copies=[]; offsets={}
    for offset in sorted(used):
        compact=(compact+7)//8*8;offsets[offset]=compact
        copies.append(f'memcpy(m->weights+{compact},w+{offset},{counts[offset]}*sizeof(float));')
        compact+=counts[offset]
    body=re.sub(r'm->weights\+(\d+)',lambda m:f'm->weights+{offsets[int(m[1])]}',body)
    prefix=prefix.replace('#include "full_ops.h"','#include "extended_ops.h"')
    prefix=prefix.replace(f'dpdf_block *blocks[{block_count}];',f'dpdf_block *blocks[{block_count}];\n dpdf_dense *dense[{len(dense)}];\n dpdf_convop *conv[{len(conv)}];')
    prefix=prefix.replace('free(m->weight_allocation);',f'for (int i=0;i<{len(dense)};++i) dpdf_dense_destroy(m->dense[i]);\nfor (int i=0;i<{len(conv)};++i) dpdf_convop_destroy(m->conv[i]);\nfree(m->weight_allocation);')
    prefix=prefix.replace('dpdf_model *dpdf_model_create(const float *w,size_t count,int tier) {',
        'dpdf_model *dpdf_model_create_config(const float *w,size_t count,int tier,int precision,unsigned mask) {\n'
        'if ((precision!=0 && precision!=8 && precision!=16) || mask>15) return NULL;\n'
        'if ((precision==8 && !dpdf_has_avx2()) || (precision==16 && !dpdf_has_fp16())) return NULL;')
    prefix=prefix.replace('malloc(count*sizeof(float)+31)',f'malloc({compact}*sizeof(float)+31)')
    prefix=prefix.replace('memcpy(m->weights,w,count*sizeof(float));','\n'.join(copies))
    # All remaining model-weight expressions in this prefix belong to block
    # construction; compact constant initialization uses m->weights+ without ().
    prefix=re.sub(r'm->blocks\[\d+\]=[^\n]+',lambda m:m[0].replace('(m->weights+','(w+'),prefix)
    pos=prefix.rindex('return m;')
    prefix=prefix[:pos]+'\n'.join(dense+conv)+'\n'+prefix[pos:]
    prefix+='''dpdf_model *dpdf_model_create(const float *w,size_t count,int tier) {
return dpdf_model_create_config(w,count,tier,0,0);
}
size_t dpdf_model_owned_bytes(const dpdf_model *m) {
if (!m) return 0;
'''
    prefix+=f'size_t bytes=sizeof(*m)+{compact}*sizeof(float)+31+dpdf_model_arena_bytes();\n'
    prefix+=f'for (int i=0;i<{block_count};++i) bytes+=dpdf_block_bytes(m->blocks[i]);\n'
    prefix+=f'for (int i=0;i<{len(dense)};++i) bytes+=dpdf_dense_bytes(m->dense[i]);\n'
    prefix+=f'for (int i=0;i<{len(conv)};++i) bytes+=dpdf_convop_bytes(m->conv[i]);\nreturn bytes;\n}}\n'
    path.write_text(prefix+'int dpdf_model_process('+body)
    manifest['extended']={'dense_contexts':len(dense),'conv_contexts':len(conv),'compact_constant_bytes':compact*4,'families':families}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Extended graph: {len(dense)} dense contexts, {len(conv)} convolution contexts, {compact*4} direct constant bytes')
    finalize(folder, symbol_prefix, extended=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--symbol-prefix', default='dpdf_model', type=validate_prefix,
                   help='C model symbol prefix (default: dpdf_model)')
    a=p.parse_args();extend(a.source,a.output,a.symbol_prefix)
