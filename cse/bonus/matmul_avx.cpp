
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <immintrin.h>
#include "utils.hpp"
#include "matmul_avx.hpp"
#include "naive.hpp"

#define a(i, j) a[(i)*lda + (j)]
#define b(i, j) b[(i)*ldb + (j)]
#define c(i, j) c[(i)*ldc + (j)]

#define min(a, b) (((a) < (b)) ? (a) : (b))

#define BLOCK_SIZE 8

/*
int main()
{

    const int N = 16; // we want it to be multilple of 4

    double *A = (double *)aligned_alloc(ALIGN, N * sizeof(double));
    double *B = (double *)aligned_alloc(ALIGN, N * sizeof(double));
    double *C = (double *)aligned_alloc(ALIGN, N * sizeof(double));
    double *D = (double *)aligned_alloc(ALIGN, N * sizeof(double));

    std::cout << "\nInitializing array...\n";

    double Ax[4][4] = {{1, 2, 3, 4},
                       {5, 6, 7, 8},
                       {9, 10, 11, 12},
                       {9, 10, 11, 12}};

    double Bx[4][4] = {{5, 2, 3, 4},
                       {6, 6, 7, 8},
                       {7, 10, 11, 12},
                       {8, 10, 11, 12}};

    std::cout << "\n Array Initialized!.\n";

    for (int i = 0; i < N; i++)
    {
        A[i] = *(&Ax[0][0] + i); // ptrA = (double *)(Ax); A[i] =  *(ptrA + i)
        B[i] = *(&Bx[0][0] + i);
    }

    naive_matmul(4, A, B, D);
    print2d(D, 4, 4);

    MY_MMult(4, 4, 4, A, 4, B, 4, C, 4);
    print2d(C, 4, 4);
}
*/

// static void do_block(int lda, int M, int N, int K, double *a, double *b, double *c)
// {
//     int ldb = lda;
//     int ldc = lda;
//     for (int i = 0; i < M; ++i)
//     {
//         for (int j = 0; j < N; ++j)
//         {
//             for (int k = 0; k < K; ++k)
//             {
//                 c(i, j) += a(i, k) * b(k, j);
//             }
//         }
//     }
// }

static void do_block(int lda, int M, int N, int K, double *a, double *b, double *c)
{
    int ldb = lda;
    int ldc = lda;
    // For each row i of A
    for (int i = 0; i < M; i += 4)
    {
        // For each column j of B
        for (int j = 0; j < N; j += 4)
        {
            AddDot4x4(K, &a(i, 0), lda, &b(0, j), ldb, &c(i, j), ldc);
        }
    }
}

void MY_MMult(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
    int i, j;

    for (i = 0; i < m; i += 4)
    {
        for (j = 0; j < n; j += 4)
        {
            AddDot4x4(k, &a(i, 0), lda, &b(0, j), ldb, &c(i, j), ldc);
        }
    }
}

void square_dgemm_blocked(int lda, double *a, double *b, double *c)
{
    int ldb = lda;
    int ldc = lda;
    // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE)
    {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE)
        {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE)
            {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                do_block(lda, M, N, K, &a(i, k), &b(k, j), &c(i, j));
            }
        }
    }
}

void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{

    __m256d ax, bx;
    __m256d c0x, c1x, c2x, c3x;
    // __m256d c0x = _mm256_set1_pd(0.0);
    // __m256d c1x = _mm256_set1_pd(0.0);
    // __m256d c2x = _mm256_set1_pd(0.0);
    // __m256d c3x = _mm256_set1_pd(0.0);

    // const int vLEN = 4; // vector Length  256 / 64 (double)

    for (int l = 0; l < k; l++)
    {
        ax = _mm256_set1_pd(a(0, l));
        bx = _mm256_load_pd(&b(l, 0));
        c0x = _mm256_fmadd_pd(ax, bx, c0x);

        ax = _mm256_set1_pd(a(1, l));
        c1x = _mm256_fmadd_pd(ax, bx, c1x);

        ax = _mm256_set1_pd(a(2, l));
        c2x = _mm256_fmadd_pd(ax, bx, c2x);

        ax = _mm256_set1_pd(a(3, l));
        c3x = _mm256_fmadd_pd(ax, bx, c3x);
    }

    _mm256_store_pd(&c(0, 0), c0x);
    _mm256_store_pd(&c(1, 0), c1x);
    _mm256_store_pd(&c(2, 0), c2x);
    _mm256_store_pd(&c(3, 0), c3x);
}

void square_dgemm(int n, double *A, double *B, double *C)
{
    // naive_matmul(n, A, B, C);
    // MY_MMult(n, n, n, A, n, B, n, C, n);
    square_dgemm_blocked(n, A, B, C);
}