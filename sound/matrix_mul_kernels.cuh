#ifndef MATRIX_MUL_KERNELS_CUH
#define MATRIX_MUL_KERNELS_CUH

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define TILE_WIDTH 16

// FP64 Matrix Multiplication Kernel
__global__ void matmul_fp64(const double *A, const double *B, double *C, int N) {
    __shared__ double sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ double sB[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    double sum = 0.0;
    
    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < N && t * TILE_WIDTH + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_WIDTH + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0;

        if (col < N && t * TILE_WIDTH + threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

// FP32 Matrix Multiplication Kernel
__global__ void matmul_fp32(const float *A, const float *B, float *C, int N) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < N && t * TILE_WIDTH + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_WIDTH + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_WIDTH + threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

// FP16 Matrix Multiplication Kernel
__global__ void matmul_fp16(const __half *A, const __half *B, __half *C, int N) {
    __shared__ __half sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ __half sB[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    __half sum = __float2half(0.0f);
    
    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < N && t * TILE_WIDTH + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_WIDTH + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = __float2half(0.0f);

        if (col < N && t * TILE_WIDTH + threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = __float2half(0.0f);

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum = __hadd(sum, __hmul(sA[threadIdx.y][k], sB[k][threadIdx.x]));

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

// Optimized FP8 Matrix Multiplication Kernel
__global__ void matmul_fp8(const __nv_fp8_e4m3 *A, const __nv_fp8_e4m3 *B, __nv_fp8_e4m3 *C, int N) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < N && t * TILE_WIDTH + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] = static_cast<float>(A[row * N + t * TILE_WIDTH + threadIdx.x]);
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_WIDTH + threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] = static_cast<float>(B[(t * TILE_WIDTH + threadIdx.y) * N + col]);
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = __nv_fp8_e4m3(sum);
}

#endif // MATRIX_MUL_KERNELS_CUH
