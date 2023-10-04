
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <immintrin.h>
#include "utils.hpp"
#include "matmul_avx.hpp"
#include "naive.hpp"

#define A(i, j) a[(j)*lda + (i)]
#define B(i, j) b[(j)*ldb + (i)]
#define C(i, j) c[(j)*ldc + (i)]
#define min(a, b) (((a) < (b)) ? (a) : (b))

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

// void MY_MMult(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
// {
//     int i, j;

//     for (i = 0; i < m; i += 4)
//     { /* Loop over the columns of C, unrolled by 4   TODO: Have changed the order change the comments as well */
//         /* Loop over the rows of C  // Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in one routine (four inner products) */
//         for (j = 0; j < n; j += 4)
//         {
//             AddDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
//         }
//     }
// }

void MY_MMult(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
    int i, j;

    for (j = 0; j < n; j += 4)
    { /* Loop over the columns of C, unrolled by 4 */
        for (i = 0; i < m; i += 4)
        { /* Loop over the rows of C */
            /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in one routine (four inner products) */

            AddDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

static inline void packA_mcxkc_d(
    int m,
    int k,
    double *XA,
    int ldXA,
    double *packA)
{
    int i, p;
    double *a_pntr[m];

    for (i = 0; i < m; i++)
    {
        a_pntr[i] = XA + ldXA * i;
    }

    for (p = 0; p < DGEMM_KC; p++)
    {
        for (i = 0; i < DGEMM_MR; i++)
        {
            if (i >= m || p >= k)
            { // corner case for zero padding, for non-divisible by 2 case
                *packA++ = 0;
            }
            else
            {
                *packA++ = *a_pntr[i]++;
            }
        }
    }
}

static inline void packB_kcxnc_d(
    int n,
    int k,
    double *XB,
    int ldXB, // ldXB is the original k
    double *packB)
{
    int j, p;
    double *b_pntr[n];

    for (j = 0; j < n; j++)
    {
        b_pntr[j] = XB + j;
    }

    for (p = 0; p < DGEMM_KC; p++)
    {
        for (j = 0; j < DGEMM_NR; j++)
        {
            if (j >= n || p >= k)
            { // corner case for zero padding, for non-divisible by 2 case
                *packB++ = 0;
            }
            else
            {
                *packB = *b_pntr[j];
                packB++;
                b_pntr[j] = b_pntr[j] + ldXB;
            }
        }
    }
}

static inline void bl_macro_kernel(
    int m,
    int n,
    int k,
    const double *packA,
    const double *packB,
    double *C,
    int ldc)
{
    int i, j;
    aux_t aux;

    for (i = 0; i < m; i += DGEMM_MR)
    { // 2-th loop around micro-kernel
        for (j = 0; j < n; j += DGEMM_NR)
        { // 1-th loop around micro-kernel
            (*bl_micro_kernel)(
                k,
                min(m - i, DGEMM_MR),
                min(n - j, DGEMM_NR),

                // based on (original)
                // &packA[i * ldc], // assumes sq matrix, otherwise use lda
                // &packB[j],       //

                // what you should use after real packing routine implmemented (1)
                &packA[i * k],
                &packB[j * k],
                &C[i * ldc + j],
                (unsigned long long)ldc,
                &aux);
        } // 1-th loop around micro-kernel
    }     // 2-th loop around micro-kernel
}

void bl_dgemm(
    int m,
    int n,
    int k,
    double *XA,
    int lda,
    double *XB,
    int ldb,
    double *C,
    int ldc)
{
    int ic, ib, jc, jb, pc, pb;
    double *packA, *packB;

    // Allocate packing buffers
    //
    // FIXME undef NOPACK when you implement packing
    //
#define NOPACK

// new change: undefine NOPACK for packing implementation
#undef NOPACK

#ifndef NOPACK
    packA = bl_malloc_aligned(DGEMM_KC, (DGEMM_MC / DGEMM_MR + 1) * DGEMM_MR, sizeof(double));
    packB = bl_malloc_aligned(DGEMM_KC, (DGEMM_NC / DGEMM_NR + 1) * DGEMM_NR, sizeof(double));
#endif
    for (ic = 0; ic < m; ic += DGEMM_MC)
    {                               // 5-th loop around micro-kernel
        ib = min(m - ic, DGEMM_MC); // for non divisible size, otherwise the depth of A and C subpanels
        for (pc = 0; pc < k; pc += DGEMM_KC)
        {                               // 4-th loop around micro-kernel
            pb = min(k - pc, DGEMM_KC); // for non divisible size, otherwise the width of the panel

#ifdef NOPACK
            packA = &XA[pc + ic * lda];
#else
            int i, j;
            for (i = 0; i < ib; i += DGEMM_MR)
            {
                packA_mcxkc_d(
                    min(ib - i, DGEMM_MR),    /* m */
                    pb,                       /* k */
                    &XA[pc + lda * (ic + i)], /* XA - start of micropanel in A */
                    k,                        /* ldXA */
                    &packA[0 * DGEMM_MC * pb + i * pb] /* packA */);
            }
#endif
            for (jc = 0; jc < n; jc += DGEMM_NC)
            { // 3-rd loop around micro-kernel
                jb = min(m - jc, DGEMM_NC);

#ifdef NOPACK
                packB = &XB[ldb * pc + jc];
#else
                for (j = 0; j < jb; j += DGEMM_NR)
                {
                    packB_kcxnc_d(
                        min(jb - j, DGEMM_NR) /* n */,
                        pb /* k */,
                        &XB[ldb * pc + jc + j] /* XB - starting row and column for this panel */,
                        n,             // should be ldXB instead /* ldXB */
                        &packB[j * pb] /* packB */
                    );
                }
#endif

                bl_macro_kernel(
                    ib,
                    jb,
                    pb,
                    packA,
                    packB,
                    &C[ic * ldc + jc],
                    ldc);
            } // End 3.rd loop around micro-kernel
        }     // End 4.th loop around micro-kernel
    }         // End 5.th loop around micro-kernel

#ifndef NOPACK
    free(packA);
    free(packB);
#endif
}

double *bl_malloc_aligned(
    int m,
    int n,
    int size)
{
    double *ptr;
    int err;

    err = posix_memalign((void **)&ptr, (size_t)GEMM_SIMD_ALIGN_SIZE, size * m * n);

    if (err)
    {
        printf("bl_malloc_aligned(): posix_memalign() failures");
        exit(1);
    }

    return ptr;
}

void AddDot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{

    __m256d ax, bx;
    __m256d c0x = _mm256_set1_pd(0.0);
    __m256d c1x = _mm256_set1_pd(0.0);
    __m256d c2x = _mm256_set1_pd(0.0);
    __m256d c3x = _mm256_set1_pd(0.0);

    // const int vLEN = 4; // vector Length  256 / 64 (double)

    for (int l = 0; l < k; l++)
    {
        ax = _mm256_set1_pd(a[0 + l]);
        bx = _mm256_load_pd(b + k * l);
        c0x = _mm256_fmadd_pd(ax, bx, c0x);

        ax = _mm256_set1_pd(a[4 + l]);
        c1x = _mm256_fmadd_pd(ax, bx, c1x);

        ax = _mm256_set1_pd(a[8 + l]);
        c2x = _mm256_fmadd_pd(ax, bx, c2x);

        ax = _mm256_set1_pd(a[12 + l]);
        c3x = _mm256_fmadd_pd(ax, bx, c3x);
    }

    _mm256_store_pd(c, c0x);           // &C[0]
    _mm256_store_pd(c + ldc, c1x);     // &C[4]
    _mm256_store_pd(c + 2 * ldc, c2x); // &C[8]
    _mm256_store_pd(c + 3 * ldc, c3x); // &C[12]
}

void square_dgemm(int n, double *A, double *B, double *C)
{
    naive_matmul(n, A, B, C);
    // MY_MMult(n, n, 4, A, n, B, n, C, n);
}