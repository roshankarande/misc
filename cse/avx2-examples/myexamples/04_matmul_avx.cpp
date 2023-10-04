
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <immintrin.h>

/* Create macros so that the matrices are stored in column-major order */

#define ALIGN 64
#define SIGNATURE(func) void (*func)(const double *, const double *, double *, int)

#define a(i, j) a[(i)*lda + (j)]
#define b(i, j) b[(i)*ldb + (j)]
#define c(i, j) c[(i)*ldc + (j)]

/* Routine for computing C = A * B + C */
void avx2_ukr_4x4(int, double *, int, double *, int, double *, int);
void matmul(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);
void print(double *a, int N);
void print2d(double *a, int lda, int m, int n);

int main()
{

    const int N = 48; // we want it to be multilple of 4

    double *A = (double *)aligned_alloc(ALIGN, N * sizeof(double));
    double *B = (double *)aligned_alloc(ALIGN, N * sizeof(double));
    double *C = (double *)aligned_alloc(ALIGN, N * sizeof(double));
    double *D = (double *)aligned_alloc(ALIGN, N * sizeof(double));

    std::cout << "\nInitializing array...\n";

    double Ax[4][8] = {
        {1, 2, 3, 4, 5, 6, 7, 8},
        {9, 10, 11, 12, 13, 14, 15, 16},
        {17, 18, 19, 20, 21, 22, 23, 24},
        {25, 26, 27, 28, 29, 30, 31, 32},
        // {33, 34, 35, 36, 37, 38, 39, 40},
        // {41, 42, 43, 44, 45, 46, 47, 48},
    };

    // double Bx[4][4] = {{5, 2, 3, 4},
    //                    {6, 6, 7, 8},
    //                    {7, 10, 11, 12},
    //                    {8, 10, 11, 12}};

    for (size_t i = 0; i < N; i++)
    {
        A[i] = *(&Ax[0][0] + i);
        B[i] = *(&Ax[0][0] + i);
    }

    // multiply_all_rows(A, B, D, 4);
    // print2d(D, 4, 4);

    print2d(A, 8, 4, 8);
    print2d(B, 4, 8, 4);

    // print2d(A + 11, 8, 3, 4);
    // print2d(B + 12, 4, 3, 4);

    // print(A, 4 * 8);
    // print(B, 8 * 4);
    matmul(4, 4, 8, A, 8, B, 4, C, 4);
    print2d(C, 4, 4, 4);
}

// int main()
// {

//     const int N = 16; // we want it to be multilple of 4

//     double *A = (double *)aligned_alloc(ALIGN, N * sizeof(double));
//     double *B = (double *)aligned_alloc(ALIGN, N * sizeof(double));
//     double *C = (double *)aligned_alloc(ALIGN, N * sizeof(double));
//     double *D = (double *)aligned_alloc(ALIGN, N * sizeof(double));

//     std::cout << "\nInitializing array...\n";

//     double Ax[4][4] = {{1, 2, 3, 4},
//                        {5, 6, 7, 8},
//                        {9, 10, 11, 12},
//                        {9, 10, 11, 12}};

//     double Bx[4][4] = {{5, 2, 3, 4},
//                        {6, 6, 7, 8},
//                        {7, 10, 11, 12},
//                        {8, 10, 11, 12}};

//     for (size_t i = 0; i < N; i++)
//     {
//         A[i] = *(&Ax[0][0] + i);
//         B[i] = *(&Bx[0][0] + i);
//     }

//     // multiply_all_rows(A, B, D, 4);
//     // print2d(D, 4, 4);

//     MY_MMult(4, 4, 4, A, 4, B, 4, C, 4);
//     print2d(C, 4, 4, 4);
// }

void matmul(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{
    int i, j;

    for (i = 0; i < m; i += 4)
    { /* Loop over the columns of C, unrolled by 4 */
        for (j = 0; j < n; j += 4)
        { /* Loop over the rows of C */
            /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in one routine (four inner products) */

            avx2_ukr_4x4(k, &a(i, 0), lda, &b(0, j), ldb, &c(i, j), ldc);
        }
    }
}

void avx2_ukr_4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc)
{

    __m256d ax, bx;
    __m256d c0x = _mm256_set1_pd(0.0);
    __m256d c1x = _mm256_set1_pd(0.0);
    __m256d c2x = _mm256_set1_pd(0.0);
    __m256d c3x = _mm256_set1_pd(0.0);

    const int vLEN = 4; // vector Length  256 / 64 (double)

    for (size_t l = 0; l < k; l++)
    {
        // ax = _mm256_set1_pd(A[0 + l]);
        // bx = _mm256_load_pd(B + k * l);
        // c0x = _mm256_fmadd_pd(ax, bx, c0x);

        // std::cout << "0" << l << "\t";
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

void print(double *A, int k)
{
    for (size_t i = 0; i < k; i++)
    {
        std::cout << A[i] << "  ";
    }

    std::cout << "\n\n";
}

void print2d(double *a, int lda, int m, int n)
{
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            std::cout << a(i, j) << "  ";
        }

        std::cout << std::endl;
    }
    std::cout << "\n\n";
}