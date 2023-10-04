#include <iostream>
#include <cstdlib>
#include <chrono>
#include <immintrin.h>
// #include <malloc.h>

#define ALIGN 64

struct Complex
{
    float r;
    float i;
};

Complex slow_add(float *A, float *B, int N);
Complex add(float *A, float *B, int N);
void print(float *A, int k);

int main()
{
    const int N = 1 << 26;

    float *A = (float *)aligned_alloc(ALIGN, N * sizeof(float));
    float *B = (float *)aligned_alloc(ALIGN, N * sizeof(float));

    std::cout << "Initializing array!";
    srand(1);

    for (size_t i = 0; i < N; i++)
    {
        float ra = (2.0f * ((float)rand()) / RAND_MAX);
        float rb = (2.0f * ((float)rand()) / RAND_MAX);

        A[i] = ra;
        B[i] = rb;
    }

    std::cout << "Array Initialized! \n";

    // Non-vectorized implementation -------------------
    std::cout << "Non vectorized \n";

    auto startTime = std::chrono::steady_clock::now();

    Complex c = slow_add(A, B, N);

    auto stopTime = std::chrono::steady_clock::now();

    double dt = (std::chrono::duration_cast<std::chrono::duration<double>>)(stopTime - startTime).count();
    std::cout << "Result : " << c.r << "\t" << c.i << std::endl;
    std::cout << "Time : " << dt << std::endl;

    // Vectorized version   ---------------
    std::cout << "Vectorized version \n";

    startTime = std::chrono::steady_clock::now();
    Complex d = add(A, B, N);
    stopTime = std::chrono::steady_clock::now();

    dt = (std::chrono::duration_cast<std::chrono::duration<double>>)(stopTime - startTime).count();
    std::cout << "Result : " << d.r << "\t" << d.i << std::endl;
    std::cout << "Time : " << dt << std::endl;

    print(A, 10);
    print(B, 10);
}

inline Complex slow_add(float *A, float *B, int N)
{
    float sumR = 0;
    float sumI = 0;

    for (size_t i = 0; i < N; i += 2)
    {
        float Ar = A[i];
        float Ai = A[i + 1];

        float Br = B[i];
        float Bi = -B[i + 1];

        float Cr = Ar * Br - Ai * Bi;
        float Ci = Ar * Bi + Ai * Br;

        sumR += Cr;
        sumI += Ci;
    }

    return Complex{sumR, sumI};
}

inline Complex add(float *A, float *B, int N)
{
    __m256 sumR = _mm256_set1_ps(0.0);
    __m256 sumI = _mm256_set1_ps(0.0);

    const __m256 conj = _mm256_set_ps(-1, 1, -1, 1, -1, 1, -1, 1);

    const __m256 *a = (__m256 *)A;
    const __m256 *b = (__m256 *)B;

    const int vectorLength = 8;
    const int n = N / vectorLength;

    for (int j = 0; j < n; j++)
    {
        __m256 cr = _mm256_mul_ps(a[j], b[j]);
        __m256 bConj = _mm256_mul_ps(b[j], conj);
        __m256 bFlip = _mm256_permute_ps(bConj, 0b10110001);
        __m256 ci = _mm256_mul_ps(a[j], bFlip);

        sumR = _mm256_add_ps(sumR, cr);
        sumI = _mm256_add_ps(sumI, ci);
    }

    // Todo get around this hack!!
    float *sr = (float *)&sumR;
    float *si = (float *)&sumI;

    float sumR2 = sr[0] + sr[1] + sr[2] + sr[3] + sr[4] + sr[5] + sr[6] + sr[7];
    float sumI2 = si[0] + si[1] + si[2] + si[3] + si[4] + si[5] + si[6] + si[7];

    return Complex{sumR2, sumI2};
}

void print(float *A, int k)
{
    for (size_t i = 0; i < k; i++)
    {
        std::cout << A[i] << "  ";
    }

    std::cout << std::endl;
}