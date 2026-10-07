"""Guarded generated-graph candidates on top of the frozen Oct7 profile.

Only apply to an isolated source snapshot. Two candidates retain scalar libm
and the generated graph's separate FP32 operations. The explicitly named
approximate candidates reuse existing fitted AVX2 nonlinearities; their
outputs change and must be rescored. No build, test or timing runs here.
"""
from pathlib import Path
import re

VARIANTS = ('graph_gru_fuse', 'graph_bias_relu',
            'graph_activations_approx', 'graph_gru_approx')
APPROXIMATE_VARIANTS = ('graph_activations_approx', 'graph_gru_approx')

WEIGHTS = {
    '5a5bb67a8619090c54dc2793536fe813fe38123b236d445f3433aafe3075529d': 16,
    'cb7248b8fccbff7b32f254ec2ed0061e604514b2ccd98d997cd081a876424e7b': 4,
}
PTR = r'\((?:m->arena|state_in)\+\d+\)'
ARENA = r'\(m->arena\+\d+\)'
COMMENT = re.compile(r'/\* \d+: (?P<op>\w+) (?P<path>\S+) \*/\n')
GRU = re.compile(r'/\* \d+: Gemm (?P<path>/model/\S+/grucell/)Gemm \*/\n'
                 r'.*?(?=\n/\* \d+: Reshape )', re.S)


def once(code, old, new):
    if code.count(old) != 1:
        raise ValueError(f'Expected one occurrence: {old[:100]!r}')
    return code.replace(old, new, 1)


def graph_identity(code):
    match = re.search(r'const char \*dpdf_model_weights_sha256\(void\) '
                      r'\{ return "([0-9a-f]{64})"; \}', code)
    if not match or match[1] not in WEIGHTS:
        raise ValueError('Expected the fitted dpdfnet2/8 Oct7 generated graph')
    blocks = WEIGHTS[match[1]]
    if (f'dpdf_block *blocks[{blocks}];' not in code or
            'dpdf_dense *dense[219];' not in code or
            'dpdf_convop *conv[21];' not in code):
        raise ValueError('Unexpected generated object counts')


def loop(destination, expression, count=256):
    return f'for (int i=0;i<{count};++i) {destination}[i]={expression};'


def body_nodes(block):
    comments = list(COMMENT.finditer(block))
    return [(item['op'], item['path'],
             block[item.end():comments[index+1].start() if index+1 < len(comments)
                   else len(block)].strip())
            for index, item in enumerate(comments)]


def dense_call(text):
    match = re.fullmatch(r'dpdf_dense_run\(m->dense\[(\d+)\],(' + PTR +
                         r'),(' + ARENA + r'),1\);', text)
    if not match:
        raise ValueError('Unexpected generated GRU affine call')
    return int(match[1]), match[2], match[3]


def split(text, source):
    targets = []
    lines = text.splitlines()
    if len(lines) != 3:
        raise ValueError('Expected three 256-float gate splits')
    for offset, line in zip((0, 256, 512), lines):
        match = re.fullmatch(r'for \(int b=0;b<1;\+\+b\) memcpy\((' + ARENA +
                             r')\+b\*256,' + re.escape(source) +
                             rf'\+b\*768\+{offset},256\*sizeof\(float\)\);', line)
        if not match:
            raise ValueError('Unexpected gate split layout')
        targets.append(match[1])
    return targets


def destination(text):
    match = re.fullmatch(r'for \(int i=0;i<256;\+\+i\) (' + ARENA + r')\[i\]=.+;', text)
    if not match:
        raise ValueError('Unexpected generated 256-wide elementwise loop')
    return match[1]


EXACT_GRU = '''
/* Private Oct7b graph helper: same scalar libm and separate FP32 operations.
 * a/b: three disjoint contiguous 256-wide gates. out may equal old, a gate
 * slice or a b gate slice; other partial overlap is outside this contract.
 * Exception trap order is not promised; no fast-math/reassociation or FMA. */
void dpdf_generated_gru256_fused(const float *a,const float *b,
                               const float *old,float *out) {
    for (int i=0;i<256;++i) {
        float reset_sum=b[i]+a[i];
        float reset=1.0f/(1.0f+expf(-reset_sum));
        float update_sum=b[256+i]+a[256+i];
        float update=1.0f/(1.0f+expf(-update_sum));
        float reset_product=b[512+i]*reset;
        float candidate_sum=a[512+i]+reset_product;
        float candidate=tanhf(candidate_sum);
        float difference=old[i]-candidate;
        float update_product=difference*update;
        out[i]=update_product+candidate;
    }
}
'''

APPROX_GRU = '''
/* Approximate only: reuse existing fitted nonlinearities without changing
 * their coefficients. All surrounding FP32 arithmetic remains separate. */
void dpdf_generated_gru256_approx_avx2(const float *a,const float *b,
                                     const float *old,float *out) {
    for (int i=0;i<256;i+=8) {
        __m256 reset=sigmoid8(_mm256_add_ps(_mm256_loadu_ps(b+i),_mm256_loadu_ps(a+i)));
        __m256 update=sigmoid8(_mm256_add_ps(_mm256_loadu_ps(b+256+i),_mm256_loadu_ps(a+256+i)));
        __m256 product=_mm256_mul_ps(_mm256_loadu_ps(b+512+i),reset);
        __m256 candidate=tanh8(_mm256_add_ps(_mm256_loadu_ps(a+512+i),product));
        __m256 difference=_mm256_sub_ps(_mm256_loadu_ps(old+i),candidate);
        __m256 result=_mm256_add_ps(_mm256_mul_ps(difference,update),candidate);
        _mm256_storeu_ps(out+i,result);
    }
}
'''

APPROX_ACTIVATIONS = '''
/* Private generated-graph candidates, separate sigmoid/tanh entry points.
 * Exact scalar tails retain the original behavior for unsupported counts. */
void dpdf_generated_sigmoid_approx_avx2(const float *x,float *out,size_t count) {
    size_t i=0;
    for (;i+7<count;i+=8) _mm256_storeu_ps(out+i,sigmoid8(_mm256_loadu_ps(x+i)));
    for (;i<count;++i) out[i]=1.0f/(1.0f+expf(-x[i]));
}
void dpdf_generated_tanh_approx_avx2(const float *x,float *out,size_t count) {
    size_t i=0;
    for (;i+7<count;i+=8) _mm256_storeu_ps(out+i,tanh8(_mm256_loadu_ps(x+i)));
    for (;i<count;++i) out[i]=tanhf(x[i]);
}
'''


def install_gru(source, approximate):
    path = source/'full_ops.c'
    code = path.read_text()
    if 'void dpdf_generated_gru256_fused(' in code:
        raise ValueError('GRU fusion already applied')
    code = once(code, '#include "full_ops.h"\n', '#include "full_ops.h"\n#include <math.h>\n')
    path.write_text(code+EXACT_GRU)
    path = source/'full_ops.h'
    header = path.read_text()
    header = once(header, 'void dpdf_axpy_scalar(float *, const float *, float, int);',
                  'void dpdf_axpy_scalar(float *, const float *, float, int);\n'
                  'void dpdf_generated_gru256_fused(const float *,const float *,const float *,float *);')
    if approximate:
        header = once(header, '#ifdef DPDF_X86_DISPATCH\n',
                      '#ifdef DPDF_X86_DISPATCH\n'
                      'void dpdf_generated_gru256_approx_avx2(const float *,const float *,const float *,float *);\n')
        path_avx = source/'avx2.c'
        avx = path_avx.read_text()
        if not all(name in avx for name in ('static inline __m256 sigmoid8(', 'static inline __m256 tanh8(')):
            raise ValueError('Expected existing fitted AVX2 activation helpers')
        path_avx.write_text(avx+APPROX_GRU)
    path.write_text(header)
    contract = Path(__file__).with_name('generated_gru_contract.c')
    (source/contract.name).write_text(contract.read_text())
    cmake = source/'CMakeLists.txt'
    cmake.write_text(cmake.read_text()+'''
# Private exact generated-cell helper: scalar libm, bits and alias/FP modes.
if(BUILD_TESTING AND DPDF_GENERATED_MODEL)
  add_executable(dpdf_generated_gru_contract generated_gru_contract.c)
  target_link_libraries(dpdf_generated_gru_contract PRIVATE dpdf_full)
  if(NOT WIN32)
    target_link_libraries(dpdf_generated_gru_contract PRIVATE m)
  endif()
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dpdf_generated_gru_contract PRIVATE -Wall -Wextra -Werror -ffp-contract=off -frounding-math)
    if(DPDF_SANITIZE)
      target_compile_options(dpdf_generated_gru_contract PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()
  endif()
  add_test(NAME generated_gru_contract COMMAND dpdf_generated_gru_contract)
  set_tests_properties(generated_gru_contract PROPERTIES TIMEOUT 30
    ENVIRONMENT "ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:handle_segv=0;UBSAN_OPTIONS=halt_on_error=1")
endif()
''')


def gru_fusion(code, approximate=False):
    paths = []
    expected_ops = ['Gemm','Split','Gemm','Split','Add','Sigmoid','Add',
                    'Sigmoid','Mul','Add','Tanh','Sub','Mul','Add']
    def replace(match):
        path = match['path']
        nodes = body_nodes(match[0])
        if [item[0] for item in nodes] != expected_ops:
            raise ValueError('Unexpected generated GRU node chain: '+path)
        if any(not item[1].startswith(path) for item in nodes):
            raise ValueError('Unexpected GRU path boundary')
        first, feature, a_buffer = dense_call(nodes[0][2])
        second, old, b_buffer = dense_call(nodes[2][2])
        if second != first+1 or not old.startswith('(state_in+'):
            raise ValueError('Unexpected GRU affine/state mapping')
        a = split(nodes[1][2], a_buffer)
        b = split(nodes[3][2], b_buffer)
        dst = [destination(item[2]) for item in nodes[4:]]
        expressions = [f'{b[0]}[i]+{a[0]}[i]',
                       f'1.0f/(1.0f+expf(-{dst[0]}[i]))',
                       f'{b[1]}[i]+{a[1]}[i]',
                       f'1.0f/(1.0f+expf(-{dst[2]}[i]))',
                       f'{b[2]}[i]*{dst[1]}[i]',
                       f'{a[2]}[i]+{dst[4]}[i]',
                       f'tanhf({dst[5]}[i])',
                       f'{old}[i]-{dst[6]}[i]',
                       f'{dst[7]}[i]*{dst[3]}[i]',
                       f'{dst[8]}[i]+{dst[6]}[i]']
        if [item[2] for item in nodes[4:]] != [loop(out, expression) for out, expression in zip(dst, expressions)]:
            raise ValueError('Unexpected GRU arithmetic/operand order: '+path)
        # The only graph value leaving the cell is its Add_3 result. Retain its
        # arena location, including the deferred whole-model state concat.
        if first not in (85, 119, 121, 182, 184):
            raise ValueError('Unexpected standalone GRU cell')
        paths.append(path)
        call = f'dpdf_generated_gru256_fused(graph_a,graph_b,{old},{dst[-1]});'
        if approximate:
            call = ('#ifdef DPDF_X86_DISPATCH\n'
                    f'    if (m->axpy==dpdf_axpy_avx2) dpdf_generated_gru256_approx_avx2(graph_a,graph_b,{old},{dst[-1]});\n'
                    '    else\n#endif\n    '+call)
        return (f'/* Oct7b fused generated GRU {path}: separate FP32 operations. */\n'
                '{\n    float graph_a[768],graph_b[768];\n'
                f'    dpdf_dense_run(m->dense[{first}],{feature},graph_a,1);\n'
                f'    dpdf_dense_run(m->dense[{second}],{old},graph_b,1);\n'
                '    '+call+'\n}')
    result, count = GRU.subn(replace, code)
    if count != 5 or len(set(paths)) != 5:
        raise ValueError(f'Expected five generated GRUs; found {count}')
    return result


BIAS_RELU = re.compile(
    r'(?P<add>/\* \d+: Add /model/[^\n]+ \*/\n)'
    r'for \(int i=0;i<(?P<count>\d+);\+\+i\) (?P<tmp>'+ARENA+r')\[i\]='
    r'(?P<input>'+ARENA+r')\[i\]\+(?P<weight>\(m->weights\+\d+\))'
    r'\[\(\(i/1\)%(?P=count)\)\*1\];\n'
    r'(?P<relu>/\* \d+: Relu /model/[^\n]+ \*/\n)'
    r'for \(int i=0;i<(?P=count);\+\+i\) (?P<out>'+ARENA+r')\[i\]='
    r'\((?P=tmp)\[i\]>0\?(?P=tmp)\[i\]:0\.0f\);')


def arena_offset(pointer):
    return int(re.search(r'\+(\d+)\)', pointer)[1])


def safe_elementwise_overlap(first, second, count):
    a, b = arena_offset(first), arena_offset(second)
    return a == b or a+count <= b or b+count <= a


def bias_relu(code):
    def replace(match):
        n = int(match['count'])
        if not safe_elementwise_overlap(match['input'], match['out'], n):
            raise ValueError('Fused bias/ReLU would have shifted input/output overlap')
        # These eight fixed bias outputs are dead after their adjacent Relu.
        weight = arena_offset(match['weight'])
        if weight not in (16, 528, 1040, 1296, 1808, 2064, 2576, 5136):
            raise ValueError('Unexpected generated bias/ReLU tensor')
        return (match['add']+match['relu']+
                f'/* Oct7b: retain the rounded addition and comparison semantics. */\n'
                f'for (int i=0;i<{n};++i) {{ float value={match["input"]}[i]+{match["weight"]}[i]; '
                f'{match["out"]}[i]=value>0?value:0.0f; }}')
    result, count = BIAS_RELU.subn(replace, code)
    if count != 8:
        raise ValueError(f'Expected eight generated bias/ReLU chains; found {count}')
    return result


def install_activations(source):
    path = source/'avx2.c'
    avx = path.read_text()
    if 'void dpdf_generated_sigmoid_approx_avx2(' in avx:
        raise ValueError('Approximate generated activations already applied')
    if not all(name in avx for name in ('static inline __m256 sigmoid8(', 'static inline __m256 tanh8(')):
        raise ValueError('Expected existing fitted AVX2 activation helpers')
    path.write_text(avx+APPROX_ACTIVATIONS)
    path = source/'full_ops.h'
    path.write_text(once(path.read_text(), '#ifdef DPDF_X86_DISPATCH\n',
                    '#ifdef DPDF_X86_DISPATCH\n'
                    'void dpdf_generated_sigmoid_approx_avx2(const float *,float *,size_t);\n'
                    'void dpdf_generated_tanh_approx_avx2(const float *,float *,size_t);\n'))


def approximate_activations(code, fused=False):
    sigmoid = re.compile(r'for \(int i=0;i<(?P<count>\d+);\+\+i\) (?P<out>'+ARENA+r')\[i\]='
                         r'1\.0f/\(1\.0f\+expf\(-(?P<input>'+ARENA+r')\[i\]\)\);')
    tanh = re.compile(r'for \(int i=0;i<(?P<count>\d+);\+\+i\) (?P<out>'+ARENA+r')\[i\]='
                      r'tanhf\((?P<input>'+ARENA+r')\[i\]\);')
    def replace(kind, match):
        count = int(match['count'])
        if not safe_elementwise_overlap(match['input'], match['out'], count):
            raise ValueError('Approximate activation would have shifted overlap')
        return ('#ifdef DPDF_X86_DISPATCH\n'
                f'if (m->axpy==dpdf_axpy_avx2) dpdf_generated_{kind}_approx_avx2({match["input"]},{match["out"]},{count});\n'
                'else\n#endif\n'+match[0])
    code, sig_count = sigmoid.subn(lambda match: replace('sigmoid', match), code)
    code, tanh_count = tanh.subn(lambda match: replace('tanh', match), code)
    expected = (1, 1) if fused else (11, 6)
    if (sig_count, tanh_count) != expected:
        raise ValueError(f'Unexpected generated activation counts {(sig_count, tanh_count)} != {expected}')
    return code


def apply(name, source: Path):
    if name not in VARIANTS:
        raise ValueError(name)
    source = Path(source)
    if 'oct7' in source.parts and 'scratch' in source.parts:
        raise ValueError('Never mutate the preserved Oct7 baseline')
    path = source/'generated_model.c'
    original = path.read_text()
    graph_identity(original)
    if name == 'graph_bias_relu':
        code = bias_relu(original)
    elif name in ('graph_gru_fuse', 'graph_gru_approx'):
        approximate = name == 'graph_gru_approx'
        code = gru_fusion(original, approximate)
        if approximate:
            code = approximate_activations(code, fused=True)
        install_gru(source, approximate)
        if approximate:
            install_activations(source)
    else:
        code = approximate_activations(original)
        install_activations(source)
    path.write_text(code)
