
init:
	make -p bin

openmp: openmp/benchmark.cpp openmp/naive.cpp
	g++ -Wall -std=c++11 -fopenmp -D_USE_OPENBLAS -O3 -L/usr/lib  -lopenblas   -I /usr/include/openblas -march=native -mavx2 -mfma -funroll-loops openmp/benchmark.cpp openmp/naive.cpp -lopenblas -o bin/matmul-openmp
	./bin/matmul-openmp

avx: avx/avx2.cpp
	g++ -Wall  -std=c++11 -mavx2 -mfma avx/avx2.cpp -o bin/matmul-avx
	./bin/matmul-avx

run-openmp: init openmp
	./bin/matmul-openmp

run-avx: init avx
	./bin/matmul-avx

.PHONY: clean

clean:
	rm -rf bin/*