
openmp: 
	g++ -Wall -std=c++11 -fopenmp -D_USE_OPENBLAS -O3 -L/usr/lib  -lopenblas   -I /usr/include/openblas -march=native -mavx2 -mfma -funroll-loops benchmark.cpp naive.cpp -lopenblas -o matmul

run: clean openmp
	./matmul

.PHONY: clean

clean:
	rm -rf matmul