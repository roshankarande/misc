#include <iostream>
#include <cstdlib>
#include <chrono>
#include <immintrin.h>
using namespace std;

int main()
{

    __m256d a = _mm256_set_pd(4.0, 3.0, 2.0, 1.0);
    auto b = _mm256_set1_pd(2.0);
    auto c = _mm256_setzero_pd();

    auto d = _mm256_add_pd(a, b);

    cout << d[0] << d[1] << d[2] << d[3] << endl;
}