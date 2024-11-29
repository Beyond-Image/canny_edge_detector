#include <iostream>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(int* A, int* B, int* C, int N) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < N) {
        C[index] = A[index] + B[index];
    }
}

int main() {
    const int N = 10;
    int A[N], B[N], C[N];

    // Initialize input arrays
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    int* d_A, * d_B, * d_C;

    // Allocate device memory
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with one block and N threads
    vector_add_kernel <<<1, N >>> (d_A, d_B, d_C, N);

    // Copy result from device to host
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Resulting vector C:\n";
    for (int i = 0; i < N; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}