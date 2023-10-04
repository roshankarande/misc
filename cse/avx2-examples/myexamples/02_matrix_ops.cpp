#include <iostream>
#include <cstdlib>
#include <chrono>
#include <immintrin.h>
// #include <malloc.h>

#define ALIGN 64
#define SIGNATURE(func) void (*func)(const double *, const double *, double *, int)

#define A(i, j) A[N * i + j]
#define B(i, j) B[N * i + j]
#define C(i, j) C[N * i + j]

void run(const double *A, const double *B, double *C, int N, SIGNATURE(non_vectorized), SIGNATURE(vectorized));
void print(double *A, int N);
void print2d(double *A, int M, int N); // if the matrix is 5x4 you pass M = 5, N = 4

void slow_add(const double *A, const double *B, double *C, int N);
void add(const double *A, const double *B, double *C, int N);

void slow_elwise_multiply(const double *A, const double *B, double *C, int N);
void elwise_multiply(const double *A, const double *B, double *C, int N);

void slow_multiply(const double *A, const double *B, double *C, int N);
void fast_multiply(const double *A, const double *B, double *C, int N);
void multiply(const double *A, const double *B, double *C, int N);
void multiply_1_row(const double *A, const double *B, double *C, int N);
void multiply_all_rows(const double *A, const double *B, double *C, int N);
void multiply_all_reduced(const double *A, const double *B, double *C, int N);
void multiply_all_rows_elaborated(const double *A, const double *B, double *C, int N);

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

    for (size_t i = 0; i < N; i++)
    {
        A[i] = *(&Ax[0][0] + i);
        B[i] = *(&Bx[0][0] + i);
    }

    print2d(A, 4, 4);
    print2d(B, 4, 4);

    std::cout << "Array Initialized! \n";

    slow_multiply(A, B, C, 4);

    // print(ptrC, 16);
    print2d(C, 4, 4);

    // multiply_all_rows(A, B, D, 4);
    // print2d(D, 4, 4);

    multiply_all_rows_elaborated(A, B, D, 4);
    // multiply_all_reduced(A, B, D, 4);
    print2d(D, 4, 4);
}

void multiply_1_row(const double *A, const double *B, double *C, int N)
{
    __m256d ax, bx;
    __m256d c0x = _mm256_set1_pd(0.0);
    __m256d c1x = _mm256_set1_pd(0.0);

    const int vLEN = 4; // vector Length  256 / 64 (double)
    const int n = N / vLEN;

    ax = _mm256_set1_pd(A[0]);
    bx = _mm256_load_pd(B + 0);
    c0x = _mm256_fmadd_pd(ax, bx, c0x);

    ax = _mm256_set1_pd(A[1]);
    bx = _mm256_load_pd(B + 4);
    c0x = _mm256_fmadd_pd(ax, bx, c0x);

    ax = _mm256_set1_pd(A[2]);
    bx = _mm256_load_pd(B + 8);
    c0x = _mm256_fmadd_pd(ax, bx, c0x);

    ax = _mm256_set1_pd(A[3]);
    bx = _mm256_load_pd(B + 12);
    c0x = _mm256_fmadd_pd(ax, bx, c0x);

    _mm256_store_pd(&C[0], c0x);
}

void multiply_all_rows(const double *A, const double *B, double *C, int N)
{
    __m256d ax, bx;
    // __m256d c0x, c1x, c2x, c3x;
    __m256d c0x = _mm256_set1_pd(0.0);
    __m256d c1x = _mm256_set1_pd(0.0);
    __m256d c2x = _mm256_set1_pd(0.0);
    __m256d c3x = _mm256_set1_pd(0.0);

    const int vLEN = 4; // vector Length  256 / 64 (double)
    const int n = N / vLEN;

    // ---------- row 1 ----------------------

    ax = _mm256_set1_pd(A[0]);
    bx = _mm256_load_pd(B + 0);
    c0x = _mm256_fmadd_pd(ax, bx, c0x);

    ax = _mm256_set1_pd(A[1]);
    bx = _mm256_load_pd(B + 4);
    c0x = _mm256_fmadd_pd(ax, bx, c0x);

    ax = _mm256_set1_pd(A[2]);
    bx = _mm256_load_pd(B + 8);
    c0x = _mm256_fmadd_pd(ax, bx, c0x);

    ax = _mm256_set1_pd(A[3]);
    bx = _mm256_load_pd(B + 12);
    c0x = _mm256_fmadd_pd(ax, bx, c0x);

    // ---------- row 2 ----------------------

    ax = _mm256_set1_pd(A[4]);
    bx = _mm256_load_pd(B);
    c1x = _mm256_fmadd_pd(ax, bx, c1x);

    ax = _mm256_set1_pd(A[5]);
    bx = _mm256_load_pd(B + 4);
    c1x = _mm256_fmadd_pd(ax, bx, c1x);

    ax = _mm256_set1_pd(A[6]);
    bx = _mm256_load_pd(B + 8);
    c1x = _mm256_fmadd_pd(ax, bx, c1x);

    ax = _mm256_set1_pd(A[7]);
    bx = _mm256_load_pd(B + 12);
    c1x = _mm256_fmadd_pd(ax, bx, c1x);

    // ---------- row 3 ----------------------

    ax = _mm256_set1_pd(A[8]);
    bx = _mm256_load_pd(B);
    c2x = _mm256_fmadd_pd(ax, bx, c2x);

    ax = _mm256_set1_pd(A[9]);
    bx = _mm256_load_pd(B + 4);
    c2x = _mm256_fmadd_pd(ax, bx, c2x);

    ax = _mm256_set1_pd(A[10]);
    bx = _mm256_load_pd(B + 8);
    c2x = _mm256_fmadd_pd(ax, bx, c2x);

    ax = _mm256_set1_pd(A[11]);
    bx = _mm256_load_pd(B + 12);
    c2x = _mm256_fmadd_pd(ax, bx, c2x);

    // ---------- row 4 ----------------------

    ax = _mm256_set1_pd(A[12]);
    bx = _mm256_load_pd(B);
    c3x = _mm256_fmadd_pd(ax, bx, c3x);

    ax = _mm256_set1_pd(A[13]);
    bx = _mm256_load_pd(B + 4);
    c3x = _mm256_fmadd_pd(ax, bx, c3x);

    ax = _mm256_set1_pd(A[14]);
    bx = _mm256_load_pd(B + 8);
    c3x = _mm256_fmadd_pd(ax, bx, c3x);

    ax = _mm256_set1_pd(A[15]);
    bx = _mm256_load_pd(B + 12);
    c3x = _mm256_fmadd_pd(ax, bx, c3x);

    _mm256_store_pd(&C[0], c0x);
    _mm256_store_pd(&C[4], c1x);
    _mm256_store_pd(&C[8], c2x);
    _mm256_store_pd(&C[12], c3x);
}

void multiply_all_rows_elaborated(const double *A, const double *B, double *C, int N)
{
    __m256d ax, bx;
    // __m256d c0x, c1x, c2x, c3x;
    __m256d c0x = _mm256_set1_pd(0.0);
    __m256d c1x = _mm256_set1_pd(0.0);
    __m256d c2x = _mm256_set1_pd(0.0);
    __m256d c3x = _mm256_set1_pd(0.0);

    const int vLEN = 4; // vector Length  256 / 64 (double)
    const int n = N / vLEN;

    // ---------- row 1 ----------------------

    ax = _mm256_set1_pd(A(0, 0));
    bx = _mm256_load_pd(&B(0, 0)); // this will load b00, b01, b02, b03
    c0x = _mm256_fmadd_pd(ax, bx, c0x);

    ax = _mm256_set1_pd(A(0, 1));
    bx = _mm256_load_pd(&B(1, 0)); // this will load b10, b11, b12, b13
    c0x = _mm256_fmadd_pd(ax, bx, c0x);

    ax = _mm256_set1_pd(A(0, 2));
    bx = _mm256_load_pd(&B(2, 0));
    c0x = _mm256_fmadd_pd(ax, bx, c0x);

    ax = _mm256_set1_pd(A(0, 3));
    bx = _mm256_load_pd(&B(3, 0));
    c0x = _mm256_fmadd_pd(ax, bx, c0x);

    // ---------- row 2 ----------------------

    ax = _mm256_set1_pd(A(1, 0));
    bx = _mm256_load_pd(B);
    c1x = _mm256_fmadd_pd(ax, bx, c1x);

    ax = _mm256_set1_pd(A(1, 1));
    bx = _mm256_load_pd(B + 4);
    c1x = _mm256_fmadd_pd(ax, bx, c1x);

    ax = _mm256_set1_pd(A[6]);
    bx = _mm256_load_pd(B + 8);
    c1x = _mm256_fmadd_pd(ax, bx, c1x);

    ax = _mm256_set1_pd(A[7]);
    bx = _mm256_load_pd(B + 12);
    c1x = _mm256_fmadd_pd(ax, bx, c1x);

    // ---------- row 3 ----------------------

    ax = _mm256_set1_pd(A[8]);
    bx = _mm256_load_pd(B);
    c2x = _mm256_fmadd_pd(ax, bx, c2x);

    ax = _mm256_set1_pd(A[9]);
    bx = _mm256_load_pd(B + 4);
    c2x = _mm256_fmadd_pd(ax, bx, c2x);

    ax = _mm256_set1_pd(A[10]);
    bx = _mm256_load_pd(B + 8);
    c2x = _mm256_fmadd_pd(ax, bx, c2x);

    ax = _mm256_set1_pd(A[11]);
    bx = _mm256_load_pd(B + 12);
    c2x = _mm256_fmadd_pd(ax, bx, c2x);

    // ---------- row 4 ----------------------

    ax = _mm256_set1_pd(A[12]);
    bx = _mm256_load_pd(B);
    c3x = _mm256_fmadd_pd(ax, bx, c3x);

    ax = _mm256_set1_pd(A[13]);
    bx = _mm256_load_pd(B + 4);
    c3x = _mm256_fmadd_pd(ax, bx, c3x);

    ax = _mm256_set1_pd(A[14]);
    bx = _mm256_load_pd(B + 8);
    c3x = _mm256_fmadd_pd(ax, bx, c3x);

    ax = _mm256_set1_pd(A[15]);
    bx = _mm256_load_pd(B + 12);
    c3x = _mm256_fmadd_pd(ax, bx, c3x);

    _mm256_store_pd(&C[0], c0x);
    _mm256_store_pd(&C[4], c1x);
    _mm256_store_pd(&C[8], c2x);
    _mm256_store_pd(&C[12], c3x);
}

void multiply_all_reduced(const double *A, const double *B, double *C, int N)
{
    __m256d ax, bx;
    __m256d c0x = _mm256_set1_pd(0.0);
    __m256d c1x = _mm256_set1_pd(0.0);
    __m256d c2x = _mm256_set1_pd(0.0);
    __m256d c3x = _mm256_set1_pd(0.0);

    const int vLEN = 4; // vector Length  256 / 64 (double)
    const int n = N / vLEN;

    //  k == number of additions that have to be done for each row to computed.. 4 in this case
    // We will take 1 block from each of the explict loops returned
    // ---------- row 1 ----------------------

    for (size_t k = 0; k < N; k++)
    {
        ax = _mm256_set1_pd(A[0 + k]);
        bx = _mm256_load_pd(B + N * k);
        c0x = _mm256_fmadd_pd(ax, bx, c0x);

        ax = _mm256_set1_pd(A[4 + k]);
        // bx = _mm256_load_pd(B + N * k);
        c1x = _mm256_fmadd_pd(ax, bx, c1x);

        ax = _mm256_set1_pd(A[8 + k]);
        // bx = _mm256_load_pd(B + N * k);
        c2x = _mm256_fmadd_pd(ax, bx, c2x);

        ax = _mm256_set1_pd(A[12 + k]);
        // bx = _mm256_load_pd(B + N * k);
        c3x = _mm256_fmadd_pd(ax, bx, c3x);
    }

    _mm256_store_pd(&C[0], c0x);
    _mm256_store_pd(&C[4], c1x);
    _mm256_store_pd(&C[8], c2x);
    _mm256_store_pd(&C[12], c3x);
}

// void multiply(const double *A, const double *B, double *C, int N)
// {
//     __m256d ax, bx;
//     __m256d c0x = _mm256_set1_pd(0.0);
//     __m256d c1x = _mm256_set1_pd(0.0);

//     const int vLEN = 4; // vector Length  256 / 64 (double)
//     const int n = N / vLEN;

//     ax = _mm256_set1_pd(A[0]);
//     bx = _mm256_load_pd(A);
//     // c0x = _mm256_mul_pd(ax, bx);
//     c0x = _mm256_fmadd_pd(ax, bx, c0x);

//     ax = _mm256_set1_pd(A[4]);
//     bx = _mm256_load_pd(&A[4]);
//     c1x = _mm256_fmadd_pd(ax, bx, c1x); // row 2

//     _mm256_store_pd(&C[0], c0x);
//     _mm256_store_pd(&C[4], c1x);
// }

// int main()
// {
//     const int N = 1 << 26; // we want it to be multilple of 4

//     double *A = (double *)aligned_alloc(ALIGN, N * sizeof(double));
//     double *B = (double *)aligned_alloc(ALIGN, N * sizeof(double));
//     double *C = (double *)aligned_alloc(ALIGN, N * sizeof(double));

//     std::cout << "\nInitializing array...\n";
//     srand(0);

//     for (size_t i = 0; i < N; i++)
//     {
//         double ra = (2.0f * ((double)rand()) / RAND_MAX) - 1.0f;
//         double rb = (2.0f * ((double)rand()) / RAND_MAX) - 1.0f;

//         A[i] = ra;
//         B[i] = rb;
//     }

//     std::cout << "Array Initialized! \n";

//     run(A, B, C, N, &slow_add, &add);
//     run(A, B, C, N, &slow_elwise_multiply, &elwise_multiply);
// }

void slow_add(const double *A, const double *B, double *C, int N)
{
    for (size_t i = 0; i < N; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

void add(const double *A, const double *B, double *C, int N)
{

    const int vLEN = 4; // vector Length  256 / 64 (double)
    const int n = N / vLEN;

    for (int j = 0; j < N; j += vLEN)
    {

        __m256d a = _mm256_load_pd(A + j);
        __m256d b = _mm256_load_pd(B + j);
        __m256d sum = _mm256_add_pd(a, b);
        _mm256_store_pd(C + j, sum);
    }
}

void slow_elwise_multiply(const double *A, const double *B, double *C, int N)
{
    for (size_t i = 0; i < N; ++i)
    {
        C[i] = A[i] * B[i];
    }
}

void elwise_multiply(const double *A, const double *B, double *C, int N)
{
    const int vLEN = 4; // vector Length  256 / 64 (double)
    const int n = N / vLEN;

    for (int j = 0; j < N; j += vLEN)
    {

        __m256d a = _mm256_load_pd(A + j);
        __m256d b = _mm256_load_pd(B + j);
        __m256d mul = _mm256_mul_pd(a, b);
        _mm256_store_pd(C + j, mul);
    }
}

void slow_multiply(const double *A, const double *B, double *C, int N)
{

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            for (size_t k = 0; k < N; k++)
            {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
}

// void fast_multiply(const double *A, const double *B, double *C, int N)
// {
//     __m256d ax, bx;
//     __m256d c0x = _mm256_set1_pd(0.0);
//     __m256d c1x = _mm256_set1_pd(0.0);

//     int k = N;
//     int lda = N;
//     int ldb = N;
//     int ldc = N;

//     int ldgemm_mr = 0;
//     for (size_t l = 0; l < k; l++)
//     {
//         ldgemm_mr = l * lda;
//         register double aval = A[ldgemm_mr];
//         ax = _mm256_set1_pd(aval);
//         bx = _mm256_load_pd(B + l * ldb);

//         c0x = _mm256_fmadd_pd(ax, bx, c0x); // row 1

//         aval = A[1 + ldgemm_mr];
//         ax = _mm256_set1_pd(aval);
//         c0x = _mm256_fmadd_pd(ax, bx, c0x); // row 2
//     }

//     _mm256_store_pd(&C[0], c0x);
//     _mm256_store_pd(&C[1], c1x);
// }

void multiply(const double *A, const double *B, double *C, int N);

void run(const double *A, const double *B, double *C, int N, SIGNATURE(non_vectorized), SIGNATURE(vectorized))
{
    // Non-vectorized implementation -------------------
    std::cout << "\nNon vectorized \n";

    auto startTime = std::chrono::steady_clock::now();
    non_vectorized(A, B, C, N);
    auto stopTime = std::chrono::steady_clock::now();

    double dt = (std::chrono::duration_cast<std::chrono::duration<double>>)(stopTime - startTime).count();
    std::cout << "Time : " << dt << std::endl;

    print(C, 10);

    // Vectorized implementation   ---------------
    std::cout << "\nVectorized version \n";

    startTime = std::chrono::steady_clock::now();
    vectorized(A, B, C, N);
    stopTime = std::chrono::steady_clock::now();

    dt = (std::chrono::duration_cast<std::chrono::duration<double>>)(stopTime - startTime).count();
    std::cout << "Time : " << dt << std::endl;

    print(C, 10);
}

void print(double *A, int k)
{
    for (size_t i = 0; i < k; i++)
    {
        std::cout << A[i] << "  ";
    }

    std::cout << "\n\n";
}

void print2d(double *A, int M, int N)
{
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            std::cout << *(A + M * i + j) << "  ";
        }

        std::cout << std::endl;
    }
    std::cout << "\n\n";
}