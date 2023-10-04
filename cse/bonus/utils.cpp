#include <iostream>
#include <chrono>
#include "utils.hpp"

void print(double *A, int k)
{
    for (int i = 0; i < k; i++)
    {
        std::cout << A[i] << "  ";
    }

    std::cout << "\n\n";
}

void print2d(double *A, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << *(A + M * i + j) << "  ";
        }

        std::cout << std::endl;
    }
    std::cout << "\n\n";
}

void run(const double *A, const double *B, double *C, int N, SIGNATURE(f1), SIGNATURE(f2))
{
    // Non-vectorized implementation -------------------
    std::cout << "\n f1  \n";

    auto startTime = std::chrono::steady_clock::now();
    f1(A, B, C, N);
    auto stopTime = std::chrono::steady_clock::now();

    double dt = (std::chrono::duration_cast<std::chrono::duration<double>>)(stopTime - startTime).count();
    std::cout << "Time : " << dt << std::endl;

    print(C, 10);

    // Vectorized implementation   ---------------
    std::cout << "\n f2 \n";

    startTime = std::chrono::steady_clock::now();
    f2(A, B, C, N);
    stopTime = std::chrono::steady_clock::now();

    dt = (std::chrono::duration_cast<std::chrono::duration<double>>)(stopTime - startTime).count();
    std::cout << "Time : " << dt << std::endl;

    print(C, 10);
}
