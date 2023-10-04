#!/bin/bash

# p=02-simd
# p=05-matmul
# p=01_basic
# p=02_matrix_ops
# p=03_try
p=04_matmul_avx

make clean
make $p
./bin/$p

# g++ -std=c++11 examples/01-hello.c -o bin/01-hello -mavx -S