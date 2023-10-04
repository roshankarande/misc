CC=g++ -std=c++11 -mavx2 -mfma # -O3

all: 01 04

01: examples/01-hello.c
	$(CC) examples/01-hello.c -o bin/01-hello

04: examples/04-dot-product.c
	$(CC) examples/04-dot-product.c -o bin/04-dot-product

02-simd: examples/02-simd.cpp
	$(CC) examples/02-simd.cpp -o bin/02-simd

05-matmul: examples/05-matmul.cpp
	$(CC) examples/05-matmul.cpp -o bin/05-matmul

01_basic: myexamples/01_basic.cpp
	$(CC) myexamples/01_basic.cpp -o bin/01_basic

02_matrix_ops: myexamples/02_matrix_ops.cpp
	$(CC) myexamples/02_matrix_ops.cpp -o bin/02_matrix_ops

03_try: myexamples/03_try.cpp
	$(CC) myexamples/03_try.cpp -o bin/03_try

04_matmul_avx: myexamples/04_matmul_avx.cpp
	$(CC) myexamples/04_matmul_avx.cpp -o bin/04_matmul_avx	


.PHONY: clean

clean:
	rm -rf bin/*