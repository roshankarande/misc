#include <iostream>
#include <cstdlib>
#include <chrono>
#include <immintrin.h>

#define ALIGN 64

void complex_square(double *a, double *x, double *y, int n)
{
    for (int i = 0; i < n; i += 2)
    {
        y[i] = a[0] + x[i] * x[i] - x[i + 1] * x[i + 1];
        y[i + 1] = a[1] + 2.0 * x[i] * x[i + 1];
    }
}

void complex_square_fma(double *a, double *x, double *y, int n)
{
    __m256d re, im, a_re, a_im, two;
    two = _mm256_set1_pd(2.0);
    a_re = _mm256_set1_pd(a[0]);
    a_im = _mm256_set1_pd(a[1]);
    for (int i = 0; i < n; i += 2)
    {
        auto x_re = _mm256_load_pd(x + i);
        auto x_im = _mm256_load_pd(x + i + 1);
        re = _mm256_fmadd_pd(x_re, x_re, a_re);
        re = _mm256_fnmadd_pd(x_im, x_im, re);
        im = _mm256_mul_pd(two, x_re);
        im = _mm256_fmadd_pd(im, x_im, a_im);
        _mm256_store_pd(y + i, re);
        _mm256_store_pd(y + i + 1, im);
    }
}

void print(double *A, int k)
{
    for (size_t i = 0; i < k; i++)
    {
        std::cout << A[i] << "  ";
    }

    std::cout << std::endl;
}

int main()
{
    const int N = 1 << 12; // we want it to be multilple of 4

    double *A = (double *)aligned_alloc(ALIGN, N * sizeof(double));
    double *B = (double *)aligned_alloc(ALIGN, N * sizeof(double));
    double *C = (double *)aligned_alloc(ALIGN, N * sizeof(double));

    std::cout << "\nInitializing array...\n";

    for (size_t i = 0; i < N; i++)
    {
        double ra = (2.0f * ((double)rand()) / RAND_MAX) - 1.0f;
        double rb = (2.0f * ((double)rand()) / RAND_MAX) - 1.0f;

        A[i] = ra;
        B[i] = rb;
    }

    std::cout << "Array Initialized! \n";

    complex_square(A, B, C, N);

    // print(ptrC, 16);
    print(C, N);

    complex_square_fma(A, B, C, N);
    print(C, N);
}
