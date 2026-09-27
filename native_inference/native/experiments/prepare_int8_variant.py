"""Create isolated research source trees; never change a production preset.

Run from native_inference on Linux, then configure the printed source tree with
the ordinary native CMake target. Generated libraries retain the *development*
ABI and must not be substituted for the shipped W8A8 integration library.
"""
import argparse
import hashlib
from pathlib import Path
import re
import shutil


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one source anchor: {old!r}')
    return text.replace(old, new)


def transform(code, variant):
    if variant == 'baseline':
        return code
    if variant == 'inline':
        return once(code, 'static __attribute__((noinline)) void qdot4pair(',
                    'static inline __attribute__((always_inline)) void qdot4pair(')
    if variant == 'fixed64':
        start = code.index('static __attribute__((noinline)) void qdot64(')
        end = code.index('\nstatic void qaffine_row', start)
        duplicate = code[start:end].replace('qdot64(', 'qdot64_fixed(')
        duplicate = once(duplicate, 'int k,int32_t *out)', 'int32_t *out)')
        duplicate = once(duplicate, '    const __m256i ones=',
                         '    const int k=64;\n    const __m256i ones=')
        code = code[:end]+'\n'+duplicate+code[end:]
        return once(code, 'qdot64(activation,q->packed+c*k,k,integer_sums);',
                    'if (k==64) qdot64_fixed(activation,q->packed+c*k,integer_sums);\n'
                    '        else qdot64(activation,q->packed+c*k,k,integer_sums);')
    if variant in ('unroll', 'asm4'):
        start = code.index('static __attribute__((noinline)) void qdot4pair(')
        end = code.index('static void qaffine_batch_tiled', start)
        block = code[start:end]
        if variant == 'unroll':
            block = once(block, '    for (int j=0;', '    #pragma GCC unroll 2\n    for (int j=0;')
        else:
            block = ('void qdot4pair(const int8_t *,const int8_t *,const int8_t *,int,__m256i *);\n\n')
        return code[:start]+block+code[end:]

    # Both reduced ranges allow a direct unsigned-byte / signed-byte product.
    # Worst-case adjacent-pair sums: W8A7 = 2*127*127 = 32258;
    # W7A8 = 2*254*63 = 32004. Neither saturates signed 16-bit maddubs.
    assert variant in ('u7', 'w7')
    start = code.index('dpdf_qmatrix *dpdf_qcreate')
    end = code.index('size_t dpdf_qbytes', start)
    if variant == 'w7':
        code = code[:start]+code[start:end].replace('127', '63')+code[end:]
    start = code.index('static void quantize(')
    end = code.index('#ifndef DPDF_DISABLE_QAFFINE_ROW', start)
    quant = code[start:end]
    if variant == 'u7':
        quant = quant.replace('254', '127')
    quant = once(quant, '*zp=127; return;', '*zp=0; return;')
    quant = once(quant, '        q=_mm256_sub_epi32(q,_mm256_set1_epi32(127));\n', '')
    # unsigned saturation is essential for W7A8 values above 127.
    quant = once(quant, 'p=_mm_packs_epi16(p,p);', 'p=_mm_packus_epi16(p,p);')
    code = code[:start]+quant+code[end:]
    code = code.replace('127-zp', '-zp')
    code = re.sub(r'_mm256_sign_epi8\((\w+),(\w+)\)', r'\1', code)
    code = re.sub(r'_mm256_abs_epi8\((\w+)\)', r'\1', code)
    # The remaining absolute=a aliases are eliminated by the compiler.
    code = code[code.index('#include "internal.h"'):]
    return (f'/* RESEARCH ONLY: {variant}, unsigned activations, changed quantization.\n'
            ' * This is not the production W8A8 preset. */\n'+code)


def oracle(code, variant):
    if variant not in ('u7', 'w7'):
        return code
    # The independent float-to-integer scalar oracle uses the candidate grid;
    # its arithmetic does not invoke the SIMD quantizer or packed dot kernels.
    start = code.index('static int row_contract')
    end = code.index('static int batch_contract', start)
    row = code[start:end]
    if variant == 'u7':
        row = row.replace('254', '127')
    if variant == 'w7':
        split = row.index('float ws=')
        row = row[:split]+row[split:].replace('127', '63')
    code = code[:start]+row+code[end:]
    # Existing final fixture assumes the original W8A8 grid. The independent
    # oracle above already covers extreme, zero, unaligned and tile-tail cases.
    start = code.index('    for (int k=64;k<=512;k*=2)')
    code = code[:start]+'    puts("Reduced-range scalar oracle and batch contracts passed");\n    return 0;\n}\n'
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=['baseline', 'inline', 'fixed64', 'unroll', 'asm4', 'u7', 'w7'])
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    native = Path(__file__).resolve().parents[1]
    destination = args.destination.resolve()
    if destination.is_relative_to(native):
        parser.error('Destination must be outside the source tree.')
    if destination.exists():
        parser.error('Use a fresh destination to preserve previous experiments.')
    code = (native/'int8.c').read_text()
    # Normalize line endings so Windows and Linux checkouts use the same guard.
    expected = '452c33bd36742570965042dabbef43fbb89172102d7ea30b5fe097ca2750b5d8'
    if hashlib.sha256(code.encode()).hexdigest() != expected:
        parser.error('int8.c changed since this experiment; review transforms before updating its source hash.')
    changed = transform(code, args.variant)
    shutil.copytree(native, destination, ignore=shutil.ignore_patterns('experiments', '__pycache__', 'integration'))
    (destination/'int8.c').write_text(changed)
    (destination/'int8_contract.c').write_text(oracle((native/'int8_contract.c').read_text(), args.variant))
    cmake = (destination/'CMakeLists.txt').read_text()
    cmake = cmake.replace('${CMAKE_CURRENT_SOURCE_DIR}/../models/native_blocks/erb_0.f32',
                          (native.parent/'models/native_blocks/erb_0.f32').as_posix())
    if args.variant == 'asm4':
        shutil.copyfile(native/'experiments/qdot4pair_avx2.S', destination/'qdot4pair_avx2.S')
        cmake += '\nenable_language(ASM)\ntarget_sources(dpdf_kernels PRIVATE qdot4pair_avx2.S)\n'
    (destination/'CMakeLists.txt').write_text(cmake)
    print(destination)


if __name__ == '__main__':
    main()
