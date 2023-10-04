	.file	"01-hello.c"
	.text
	.section	.rodata
.LC4:
	.string	"%f %f %f %f\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4045:
	.cfi_startproc
	endbr64
	leaq	8(%rsp), %r10
	.cfi_def_cfa 10, 0
	andq	$-32, %rsp
	pushq	-8(%r10)
	pushq	%rbp
	movq	%rsp, %rbp
	.cfi_escape 0x10,0x6,0x2,0x76,0
	pushq	%r10
	.cfi_escape 0xf,0x3,0x76,0x78,0x6
	subq	$328, %rsp
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	vmovsd	.LC0(%rip), %xmm0
	vmovsd	%xmm0, -304(%rbp)
	vmovsd	.LC1(%rip), %xmm0
	vmovsd	%xmm0, -296(%rbp)
	vmovsd	.LC2(%rip), %xmm0
	vmovsd	%xmm0, -288(%rbp)
	vmovsd	.LC3(%rip), %xmm0
	vmovsd	%xmm0, -280(%rbp)
	vmovsd	-304(%rbp), %xmm0
	vmovsd	-296(%rbp), %xmm1
	vunpcklpd	%xmm0, %xmm1, %xmm1
	vmovsd	-288(%rbp), %xmm0
	vmovsd	-280(%rbp), %xmm2
	vunpcklpd	%xmm0, %xmm2, %xmm0
	vinsertf128	$0x1, %xmm1, %ymm0, %ymm0
	vmovapd	%ymm0, -272(%rbp)
	vmovapd	-272(%rbp), %ymm0
	vmovapd	%ymm0, -240(%rbp)
	vmovapd	-272(%rbp), %ymm0
	vmovapd	%ymm0, -144(%rbp)
	vmovapd	-240(%rbp), %ymm0
	vmovapd	%ymm0, -112(%rbp)
	vmovapd	-144(%rbp), %ymm0
	vaddpd	-112(%rbp), %ymm0, %ymm0
	vmovapd	%ymm0, -208(%rbp)
	leaq	-80(%rbp), %rax
	movq	%rax, -312(%rbp)
	vmovapd	-208(%rbp), %ymm0
	vmovapd	%ymm0, -176(%rbp)
	movq	-312(%rbp), %rax
	vmovapd	-176(%rbp), %ymm0
	vmovapd	%ymm0, (%rax)
	nop
	vmovsd	-56(%rbp), %xmm2
	vmovsd	-64(%rbp), %xmm1
	vmovsd	-72(%rbp), %xmm0
	movq	-80(%rbp), %rax
	vmovapd	%xmm2, %xmm3
	vmovapd	%xmm1, %xmm2
	vmovapd	%xmm0, %xmm1
	vmovq	%rax, %xmm0
	leaq	.LC4(%rip), %rdi
	movl	$4, %eax
	call	printf@PLT
	movl	$0, %eax
	movq	-24(%rbp), %rdx
	xorq	%fs:40, %rdx
	je	.L5
	call	__stack_chk_fail@PLT
.L5:
	addq	$328, %rsp
	popq	%r10
	.cfi_def_cfa 10, 0
	popq	%rbp
	leaq	-8(%r10), %rsp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4045:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC0:
	.long	0
	.long	1072693248
	.align 8
.LC1:
	.long	0
	.long	1073741824
	.align 8
.LC2:
	.long	0
	.long	1074266112
	.align 8
.LC3:
	.long	0
	.long	1074790400
	.ident	"GCC: (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	 1f - 0f
	.long	 4f - 1f
	.long	 5
0:
	.string	 "GNU"
1:
	.align 8
	.long	 0xc0000002
	.long	 3f - 2f
2:
	.long	 0x3
3:
	.align 8
4:
