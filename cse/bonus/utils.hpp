void print(double *A, int N);
void print2d(double *A, int M, int N);
#define SIGNATURE(func) void (*func)(const double *, const double *, double *, int)