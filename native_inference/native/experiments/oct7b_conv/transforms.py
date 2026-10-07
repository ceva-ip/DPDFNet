"""Exact convolution candidates exercised by both frozen Oct7 mask7 models.

Apply to isolated source snapshots only. No generation/build/test/timing is
performed automatically. The reduced-precision convop patch path is not the
accepted mask7 baseline, so these candidates target its real FP32/layout work.
"""
from pathlib import Path
import re

VARIANTS = ('conv_transpose_relu', 'conv_row_pair')
ARENA = r'\(m->arena\+\d+\)'


def once(code, old, new):
    if code.count(old) != 1:
        raise ValueError(f'Expected one occurrence: {old[:100]!r}')
    return code.replace(old, new, 1)


def install_contract(source, definition):
    contract = Path(__file__).with_name('conv_helpers_contract.c')
    (source/contract.name).write_text(contract.read_text())
    path = source/'CMakeLists.txt'
    code = path.read_text()
    if 'add_executable(dpdf_conv_helpers_contract ' not in code:
        code += '''
# Independent scalar/FMA oracle and bounded transpose/ReLU tiles.
if(BUILD_TESTING AND DPDF_GENERATED_MODEL)
  add_executable(dpdf_conv_helpers_contract conv_helpers_contract.c)
  target_link_libraries(dpdf_conv_helpers_contract PRIVATE dpdf_full)
  if(NOT WIN32)
    target_link_libraries(dpdf_conv_helpers_contract PRIVATE m)
  endif()
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dpdf_conv_helpers_contract PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
    if(DPDF_ENABLE_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
      target_compile_definitions(dpdf_conv_helpers_contract PRIVATE DPDF_X86_DISPATCH)
    endif()
    if(DPDF_SANITIZE)
      target_compile_options(dpdf_conv_helpers_contract PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()
  endif()
  add_test(NAME conv_helpers_contract COMMAND dpdf_conv_helpers_contract)
  set_tests_properties(conv_helpers_contract PROPERTIES TIMEOUT 30
    ENVIRONMENT "ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:handle_segv=0;UBSAN_OPTIONS=halt_on_error=1")
endif()
'''
    code += f'''
if(TARGET dpdf_conv_helpers_contract)
  target_compile_definitions(dpdf_conv_helpers_contract PRIVATE {definition})
endif()
'''
    path.write_text(code)


SCALAR_RELU_TRANSPOSE = '''
/* Disjoint input/output, same generated Relu conditional after bit transpose. */
void dpdf_transpose_relu_scalar(const float *x,float *y,int rows,int cols) {
    for (int r=0;r<rows;r+=8) for (int c=0;c<cols;c+=8)
        for (int i=r;i<r+8 && i<rows;++i) for (int j=c;j<c+8 && j<cols;++j) {
            float value=x[i*cols+j];
            y[j*rows+i]=value>0?value:0.0f;
        }
}
'''

RELU_TRANSPOSE = re.compile(
    r'm->transpose\((?P<input>'+ARENA+r'),(?P<tmp>'+ARENA+r'),'
    r'(?P<rows>\d+),(?P<cols>\d+)\);\n'
    r'(?P<comment>/\* \d+: Relu (?P<path>/model/[^\n]+) \*/\n)'
    r'for \(int i=0;i<(?P<count>\d+);\+\+i\) (?P<out>'+ARENA+r')\[i\]='
    r'\((?P=tmp)\[i\]>0\?(?P=tmp)\[i\]:0\.0f\);')


def transpose_relu(source):
    path = source/'generated_model.c'
    graph = path.read_text()
    if not all(item in graph for item in ('dpdf_dense *dense[219];', 'dpdf_convop *conv[21];')):
        raise ValueError('Expected dpdfnet2/8 generated pointwise graph')
    shapes = []
    def replace(match):
        rows, cols, count = (int(match[name]) for name in ('rows','cols','count'))
        if count != rows*cols or (rows,cols) not in ((40,64),(48,64),(80,64),
                                                    (96,64),(160,64),(480,64),(96,10)):
            raise ValueError('Unexpected pointwise transpose/ReLU shape')
        first, last = (int(re.search(r'\+(\d+)\)', match[name])[1]) for name in ('input','out'))
        if not (first+count <= last or last+count <= first):
            raise ValueError('Fused transpose input/output must remain disjoint')
        shapes.append((rows,cols))
        return (match['comment']+'/* Oct7b: transpose and Relu in one bounded tile store. */\n'
                '#ifdef DPDF_X86_DISPATCH\n'
                f'if (m->transpose==dpdf_transpose_avx2) dpdf_transpose_relu_avx2({match["input"]},{match["out"]},{rows},{cols});\n'
                'else\n#endif\n'
                f'dpdf_transpose_relu_scalar({match["input"]},{match["out"]},{rows},{cols});')
    graph, count = RELU_TRANSPOSE.subn(replace, graph)
    if count != 9 or shapes.count((80,64)) != 2 or shapes.count((160,64)) != 2:
        raise ValueError(f'Expected nine shared pointwise transpose/ReLU chains; found {count}')
    path_avx = source/'avx2.c'
    avx = path_avx.read_text()
    start = avx.index('void dpdf_transpose_avx2(')
    end = avx.index('void dpdf_axpy_avx2(',start)
    clone = avx[start:end].replace('void dpdf_transpose_avx2(', 'void dpdf_transpose_relu_avx2(', 1)
    for lane in range(8):
        value = f'_mm256_loadu_ps(x+(r+{lane})*cols+c)'
        clone = once(clone,value,'dpdf_transpose_relu_value_avx2('+value+')')
    clone = once(clone,'y[c*rows+r+i]=x[(r+i)*cols+c];',
                 'y[c*rows+r+i]=dpdf_transpose_relu_value_scalar(x[(r+i)*cols+c]);')
    clone = once(clone,'y[c*rows+r]=x[r*cols+c];',
                 'y[c*rows+r]=dpdf_transpose_relu_value_scalar(x[r*cols+c]);')
    helpers = '''
static inline __m256 dpdf_transpose_relu_value_avx2(__m256 value) {
    /* The generated conditional maps NaN and both zero signs to +0. */
    return _mm256_and_ps(value,_mm256_cmp_ps(value,_mm256_setzero_ps(),_CMP_GT_OQ));
}
static inline float dpdf_transpose_relu_value_scalar(float value) {
    return value>0?value:0.0f;
}
'''
    path_avx.write_text(avx+helpers+clone)
    path_full = source/'full_ops.c'
    path_full.write_text(path_full.read_text()+SCALAR_RELU_TRANSPOSE)
    path_header = source/'full_ops.h'
    header = once(path_header.read_text(),
                  'void dpdf_transpose_scalar(const float *, float *, int, int);',
                  'void dpdf_transpose_scalar(const float *, float *, int, int);\n'
                  'void dpdf_transpose_relu_scalar(const float *,float *,int,int);')
    header = once(header,'#ifdef DPDF_X86_DISPATCH\n',
                  '#ifdef DPDF_X86_DISPATCH\n'
                  'void dpdf_transpose_relu_avx2(const float *,float *,int,int);\n')
    path_header.write_text(header)
    path.write_text(graph)
    install_contract(source,'DPDF_OCT7B_TRANSPOSE_RELU')


PAIR_KERNEL = '''
/* Two outputs share input vectors; each output preserves the original
 * ic/ky reduction order and full-vector FMA versus scalar-tail distinction.
 * Used by shared DF convolutions: CI=32, KH=5, CO=5, W=96. */
int dpdf_conv_row_pair_avx2(const float *x,const float *w,const float *bias,float *y,
        int ci,int hi,int wi,int co,int ho,int wo,int kh,int kw,
        int sh,int sw,int ph,int pw,int group) {
    if (ci<=0 || co<=0 || group<=0 || ci%group || co%group || co/group<2 ||
        hi!=kh || ho!=1 || kw!=1 || sh!=1 || sw!=1 || ph || pw ||
        wi!=wo || wi<8 || kh<=0) return 0;
    int cig=ci/group,cog=co/group,k=cig*kh;
    for (int g=0;g<group;++g) {
        const float *input=x+g*cig*kh*wi;
        int oc=0;
        for (;oc+1<cog;oc+=2) {
            const float *w0=w+(g*cog+oc)*k,*w1=w0+k;
            float *out0=y+(g*cog+oc)*wo,*out1=out0+wo;
            float initial0=bias?bias[g*cog+oc]:0.0f;
            float initial1=bias?bias[g*cog+oc+1]:0.0f;
            int ow=0;
            for (;ow+31<wo;ow+=32) {
                __m256 a0=_mm256_set1_ps(initial0),a1=a0,a2=a0,a3=a0;
                __m256 b0=_mm256_set1_ps(initial1),b1=b0,b2=b0,b3=b0;
                for (int ic=0;ic<cig;++ic) for (int ky=0;ky<kh;++ky) {
                    int j=ic*kh+ky;
                    const float *p=input+j*wi+ow;
                    __m256 v0=_mm256_loadu_ps(p),v1=_mm256_loadu_ps(p+8);
                    __m256 v2=_mm256_loadu_ps(p+16),v3=_mm256_loadu_ps(p+24);
                    __m256 weight0=_mm256_set1_ps(w0[j]),weight1=_mm256_set1_ps(w1[j]);
                    a0=_mm256_fmadd_ps(weight0,v0,a0);a1=_mm256_fmadd_ps(weight0,v1,a1);
                    a2=_mm256_fmadd_ps(weight0,v2,a2);a3=_mm256_fmadd_ps(weight0,v3,a3);
                    b0=_mm256_fmadd_ps(weight1,v0,b0);b1=_mm256_fmadd_ps(weight1,v1,b1);
                    b2=_mm256_fmadd_ps(weight1,v2,b2);b3=_mm256_fmadd_ps(weight1,v3,b3);
                }
                _mm256_storeu_ps(out0+ow,a0);_mm256_storeu_ps(out0+ow+8,a1);
                _mm256_storeu_ps(out0+ow+16,a2);_mm256_storeu_ps(out0+ow+24,a3);
                _mm256_storeu_ps(out1+ow,b0);_mm256_storeu_ps(out1+ow+8,b1);
                _mm256_storeu_ps(out1+ow+16,b2);_mm256_storeu_ps(out1+ow+24,b3);
            }
            for (;ow+7<wo;ow+=8) {
                __m256 a=_mm256_set1_ps(initial0),b=_mm256_set1_ps(initial1);
                for (int ic=0;ic<cig;++ic) for (int ky=0;ky<kh;++ky) {
                    int j=ic*kh+ky;
                    __m256 value=_mm256_loadu_ps(input+j*wi+ow);
                    a=_mm256_fmadd_ps(_mm256_set1_ps(w0[j]),value,a);
                    b=_mm256_fmadd_ps(_mm256_set1_ps(w1[j]),value,b);
                }
                _mm256_storeu_ps(out0+ow,a);_mm256_storeu_ps(out1+ow,b);
            }
            for (;ow<wo;++ow) {
                float a=initial0,b=initial1;
                for (int ic=0;ic<cig;++ic) for (int ky=0;ky<kh;++ky) {
                    int j=ic*kh+ky;
                    float value=input[j*wi+ow];
                    a+=w0[j]*value;b+=w1[j]*value;
                }
                out0[ow]=a;out1[ow]=b;
            }
        }
        if (oc<cog) {
            /* Reuse the preserved odd-channel kernel for all frequency tails. */
            int handled=dpdf_conv_row_avx2(input,w+(g*cog+oc)*k,
                       bias?bias+g*cog+oc:NULL,y+(g*cog+oc)*wo,
                       cig,kh,wi,1,1,wo,kh,1,1,1,0,0,1);
            if (!handled) return 0; /* Shape guards above imply handled. */
        }
    }
    return 1;
}
'''


def row_pair(source):
    path = source/'avx2.c'
    avx = path.read_text()
    if 'int dpdf_conv_row_pair_avx2(' in avx:
        raise ValueError('Pair convolution already applied')
    if 'int dpdf_conv_row_avx2(' not in avx:
        raise ValueError('Expected preserved row convolution for odd outputs')
    path.write_text(avx+PAIR_KERNEL)
    path = source/'full_ops.h'
    header = once(path.read_text(),'int dpdf_depthwise_stride_avx2(',
                  'int dpdf_conv_row_pair_avx2(const float *,const float *,const float *,float *,\n'
                  '                      int,int,int,int,int,int,int,int,int,int,int,int,int);\n'
                  'int dpdf_depthwise_stride_avx2(')
    path.write_text(header)
    path = source/'full_ops.c'
    code = once(path.read_text(),
                '#ifdef DPDF_X86_DISPATCH\n    if (axpy==dpdf_axpy_avx2 && sw>1 &&',
                '#ifdef DPDF_X86_DISPATCH\n'
                '    if (axpy==dpdf_axpy_avx2 && dpdf_conv_row_pair_avx2(x,w,bias,y,ci,hi,wi,co,ho,wo,\n'
                '                                                    kh,kw,sh,sw,ph,pw,group)) return;\n'
                '    if (axpy==dpdf_axpy_avx2 && sw>1 &&')
    path.write_text(code)
    install_contract(source,'DPDF_OCT7B_CONV_PAIR')


def apply(name, source: Path):
    if name not in VARIANTS:
        raise ValueError(name)
    source=Path(source)
    if 'oct7' in source.parts and 'scratch' in source.parts:
        raise ValueError('Never mutate the preserved Oct7 baseline')
    if name == 'conv_transpose_relu':
        transpose_relu(source)
    else:
        row_pair(source)
