
#define ALIGN 64

#define DGEMM_MC 1920
#define DGEMM_KC 128
#define DGEMM_NC 256
#define DGEMM_MR 4
#define DGEMM_NR 4

/* Routine for computing C = A * B + C */
void AddDot4x4(int, double *, int, double *, int, double *, int);
void MY_MMult(int m, int n, int k, double *a, int lda, double *b, int ldb, double *c, int ldc);

void PackMatrixA(int, double *, int, double *);
void PackMatrixB(int, double *, int, double *);
void InnerKernel(int, int, int, double *, int, double *, int, double *, int, int);
void square_dgemm(int n, double *A, double *B, double *C);