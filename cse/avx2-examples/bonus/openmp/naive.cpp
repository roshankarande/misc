#include "naive.hpp"
#include "omp.h"

#define A(i, j) A[(i) * (n) + (j)]
#define B(i, j) B[(i) * (n) + (j)]
#define C(i, j) C[(i) * (n) + (j)]

// const char *dgemm_desc = "Naive, three-loop dgemm.";

void square_dgemm(int n, double *A, double *B, double *C)
{

    // ikj(n, A, B, C);
    ikj(n, A, B, C);
}

void ijk(int n, double *A, double *B, double *C)
{
    omp_set_num_threads(12);
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                C(i, j) += A(i, k) * B(k, j);
}

void ikj(int n, double *A, double *B, double *C)
{
    omp_set_num_threads(12);
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                C(i, j) += A(i, k) * B(k, j);
}

void jik(int n, double *A, double *B, double *C)
{
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < n; ++k)
                C(i, j) += A(i, k) * B(k, j);
}

void jki(int n, double *A, double *B, double *C)
{
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < n; ++i)
                C(i, j) += A(i, k) * B(k, j);
}

void kij(int n, double *A, double *B, double *C)
{
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                C(i, j) += A(i, k) * B(k, j);
}

void kji(int n, double *A, double *B, double *C)
{
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                C(i, j) += A(i, k) * B(k, j);
}