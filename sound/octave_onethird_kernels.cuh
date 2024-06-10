#ifndef OCTAVE_KERNELS_CUH
#define OCTAVE_KERNELS_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

// Declare the constant variable
__constant__ double k_base;

// Function to compute the center frequency for a given band
__device__ double compute_center_frequency(int band, double f_base) {
    return f_base * pow(k_base, band);
}

// FP64 (double precision)
__global__ void octave_kernel_fp64(const double* input, double* output, int N, int num_bands, double f_base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bands) {
        double center_frequency = compute_center_frequency(idx, f_base);
        double sum = 0.0;
        for (int i = 0; i < N; ++i) {
            double frequency = i * (f_base / N);
            double weight = exp(-0.5 * pow((log2(frequency / center_frequency) * 3.0), 2.0));
            sum += input[i] * weight;
        }
        output[idx] = sum;
    }
}

// FP32 (single precision)
__global__ void octave_kernel_fp32(const float* input, float* output, int N, int num_bands, float f_base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bands) {
        float center_frequency = (float)compute_center_frequency(idx, (double)f_base);
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            float frequency = i * (f_base / N);
            float weight = expf(-0.5f * powf((log2f(frequency / center_frequency) * 3.0f), 2.0f));
            sum += input[i] * weight;
        }
        output[idx] = sum;
    }
}

// FP16 (half precision)
__global__ void octave_kernel_fp16(const half* input, half* output, int N, int num_bands, half f_base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bands) {
        half center_frequency = __float2half(compute_center_frequency(idx, __half2float(f_base)));
        half sum = __float2half(0.0f);
        for (int i = 0; i < N; ++i) {
            half frequency = __float2half(i * (__half2float(f_base) / N));
            half weight = __float2half(expf(-0.5f * powf((log2f(__half2float(frequency) / __half2float(center_frequency)) * 3.0f), 2.0f)));
            sum = __hadd(sum, __hmul(input[i], weight));
        }
        output[idx] = sum;
    }
}

// FP8 (quarter precision)
__device__ __half fp8_to_fp16(const __nv_fp8_e4m3 val) {
    // Implement the conversion logic if not available in CUDA
    // Placeholder implementation
    return __half(); // Replace with actual conversion logic
}

__device__ __nv_fp8_e4m3 fp16_to_fp8(const __half val) {
    // Implement the conversion logic if not available in CUDA
    // Placeholder implementation
    return __nv_fp8_e4m3(); // Replace with actual conversion logic
}

__global__ void octave_kernel_fp8(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, int N, int num_bands, __half f_base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_bands) {
        __half center_frequency = __float2half(compute_center_frequency(idx, __half2float(f_base)));
        __half sum = __float2half(0.0f);
        for (int i = 0; i < N; ++i) {
            __half frequency = __float2half(i * (__half2float(f_base) / N));
            __half weight = __float2half(expf(-0.5f * powf((log2f(__half2float(frequency) / __half2float(center_frequency)) * 3.0f), 2.0f)));
            sum = __hadd(sum, __hmul(fp8_to_fp16(input[i]), weight));
        }
        output[idx] = fp16_to_fp8(sum);
    }
}

#endif // OCTAVE_KERNELS_CUH
