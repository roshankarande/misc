avx: 
	g++ -std=c++11 -mavx2 -mfma avx2.cpp -o matmul

run: clean avx
	./matmul

.PHONY: clean

clean:
	rm -rf matmul