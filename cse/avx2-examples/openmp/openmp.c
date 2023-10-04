#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 40

// int main(int argc, char *argv[])
// {
//     int idx;
//     float a[N];
//     float b[N];
//     float c[N];

//     for (idx = 0; idx < N; ++idx)
//     {
//         a[idx] = b[idx] = 1.0;
//     }

//     for (idx = 0; idx < N; ++idx)
//     {
//         c[idx] = a[idx] + b[idx];
//     }

//     for (int i = 0; i < N; i++)
//     {
//         printf("%d - %f \n", i, c[i]);
//     }
// }

// int main(int argc, char *argv[])
// {
//     int nthreads, tid, idx;
//     float a[N], b[N], c[N];
//     nthreads = omp_get_num_threads();
//     printf("Number of threads = %d\n", nthreads);
// #pragma omp parallel for
//     for (idx = 0; idx < N; ++idx)
//     {
//         a[idx] = b[idx] = 1.0;
//     }
// #pragma omp parallel for
//     for (idx = 0; idx < N; ++idx)
//     {
//         c[idx] = a[idx] + b[idx];
//         tid = omp_get_thread_num();
//         printf("Thread %d: c[%d]=%f\n", tid, idx, c[idx]);
//     }
// }

// export OMP_NUM_THREADS = 5
int main()
{
    omp_set_num_threads(12);
    int nthreads = omp_get_num_threads();
    printf("Max number of threads = %d\n", omp_get_max_threads());
    printf("Number of threads = %d\n", nthreads);
#pragma omp parallel
    {

        printf("Hello World... from thread = %d\n",
               omp_get_thread_num());
    }
}