CC=g++ -std=c++17 -mavx2 -mfma -fopenmp -L/usr/lib  -lopenblas   -I /usr/include/openblas -lopenblas -O3

INCLUDES += -I/usr/include/openblas


default_target: all

matmul_avx: matmul_avx.cpp
	$(CC) -o bin/matmul_avx matmul_avx.cpp utils.cpp naive.cpp

benchmark: 
	$(CC) -o bin/benchmark benchmark.cpp matmul_avx.cpp utils.cpp naive.cpp

bench:
	g++ -Wall -std=c++11 -D_USE_OPENBLAS -O3 -I/usr/include/openblas  -march=native -mavx2 -mfma -fopenmp benchmark.cpp utils.cpp naive.cpp matmul_avx.cpp -lopenblas -o bin/matmul

# for avx1 not working
bench1:
	g++ -Wall -std=c++11 -fopenmp -D_USE_OPENBLAS -O3 -L/usr/lib  -lopenblas   -I /usr/include/openblas -march=native -mavx2 -mfma -funroll-loops benchmark.cpp utils.cpp naive.cpp matmul_avx1.cpp  -lopenblas -o matmul

clean: 
	rm -rf bin/*
.PHONY : clean

all: bench
.PHONY : all

run: matmul_avx
	./bin/matmul_avx
