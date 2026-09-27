/* Research-only Linux x86-64 SysV kernel. Same packing and exact signed-byte
 * arithmetic as int8.c:qdot4pair. Not linked into the production runtime.
 * Arguments: activation, packed0, packed1, k, output. k is a positive multiple
 * of eight <=512; output holds eight vectors in tile-major, then row order. */
.intel_syntax noprefix
.text
.p2align 5
.globl qdot4pair
.hidden qdot4pair
.type qdot4pair,@function
qdot4pair:
    movsxd rax, ecx
    lea r9, [rdi+rax]
    lea r10, [r9+rax]
    lea r11, [r10+rax]
    vpxor ymm0, ymm0, ymm0
    vpxor ymm1, ymm1, ymm1
    vpxor ymm2, ymm2, ymm2
    vpxor ymm3, ymm3, ymm3
    vpxor ymm4, ymm4, ymm4
    vpxor ymm5, ymm5, ymm5
    vpxor ymm6, ymm6, ymm6
    vpxor ymm7, ymm7, ymm7
    mov eax, 0x00010001
    vmovd xmm8, eax
    vpbroadcastd ymm8, xmm8
    xor eax, eax
.macro row base, sum0, sum1
    vpbroadcastd ymm11, DWORD PTR [\base+rax]
    vpabsb ymm12, ymm11
    vpsignb ymm13, ymm9, ymm11
    vpsignb ymm14, ymm10, ymm11
    vpmaddubsw ymm13, ymm12, ymm13
    vpmaddubsw ymm14, ymm12, ymm14
    vpmaddwd ymm13, ymm13, ymm8
    vpmaddwd ymm14, ymm14, ymm8
    vpaddd \sum0, \sum0, ymm13
    vpaddd \sum1, \sum1, ymm14
.endm
.p2align 5
.Ldot:
    vmovdqu ymm9, YMMWORD PTR [rsi+rax*8]
    vmovdqu ymm10, YMMWORD PTR [rdx+rax*8]
    row rdi, ymm0, ymm4
    row r9, ymm1, ymm5
    row r10, ymm2, ymm6
    row r11, ymm3, ymm7
    add eax, 4
    cmp eax, ecx
    jl .Ldot
    vmovdqu YMMWORD PTR [r8], ymm0
    vmovdqu YMMWORD PTR [r8+32], ymm1
    vmovdqu YMMWORD PTR [r8+64], ymm2
    vmovdqu YMMWORD PTR [r8+96], ymm3
    vmovdqu YMMWORD PTR [r8+128], ymm4
    vmovdqu YMMWORD PTR [r8+160], ymm5
    vmovdqu YMMWORD PTR [r8+192], ymm6
    vmovdqu YMMWORD PTR [r8+224], ymm7
    vzeroupper
    ret
.size qdot4pair,.-qdot4pair
.section .note.GNU-stack,"",@progbits
