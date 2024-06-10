#ifndef MEL_GRAM_KERNELS_CUH
#define MEL_GRAM_KERNELS_CUH

#include <cuda_fp16.h>
#include <math.h>
#include <cuda_fp8.h>  // Include CUDA FP8 header if available

// Define the number of mel filters
#define NUM_MEL_FILTERS 40

// Convert __nv_fp8_e4m3 to float (simulation)
__device__ float fp8_to_float(__nv_fp8_e4m3 x) {
    // Simulate conversion from FP8 to float
    return __half2float(*reinterpret_cast<__half*>(&x));
}

// Convert float to __nv_fp8_e4m3 (simulation)
__device__ __nv_fp8_e4m3 float_to_fp8(float x) {
    // Simulate conversion from float to FP8
    __half temp = __float2half(x);
    return *reinterpret_cast<__nv_fp8_e4m3*>(&temp);
}

// Compute the mel scale value
__device__ __host__ float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

// Compute the frequency value from the mel scale value
__device__ __host__ float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// Generate mel filter bank
void generate_mel_filter_bank(float* mel_filters, int num_filters, int num_fft, int sample_rate) {
    float low_freq_mel = hz_to_mel(0);
    float high_freq_mel = hz_to_mel(sample_rate / 2);
    float mel_step = (high_freq_mel - low_freq_mel) / (num_filters + 1);

    float mel_points[NUM_MEL_FILTERS + 2];
    for (int i = 0; i < NUM_MEL_FILTERS + 2; i++) {
        mel_points[i] = mel_to_hz(low_freq_mel + i * mel_step);
    }

    for (int i = 0; i < NUM_MEL_FILTERS; i++) {
        for (int j = 0; j < num_fft / 2 + 1; j++) {
            float freq = (float)j * sample_rate / num_fft;
            if (freq >= mel_points[i] && freq <= mel_points[i + 1]) {
                mel_filters[i * (num_fft / 2 + 1) + j] = (freq - mel_points[i]) / (mel_points[i + 1] - mel_points[i]);
            } else if (freq >= mel_points[i + 1] && freq <= mel_points[i + 2]) {
                mel_filters[i * (num_fft / 2 + 1) + j] = (mel_points[i + 2] - freq) / (mel_points[i + 2] - mel_points[i + 1]);
            } else {
                mel_filters[i * (num_fft / 2 + 1) + j] = 0.0f;
            }
        }
    }
}

// Kernels for different precisions using common function
template <typename T, typename U>
__device__ T apply_mel_filter_and_log(const T* input, int idx, int num_features, int num_frames, const U* mel_filters, int num_fft) {
    int mel_idx = idx / num_frames;
    int frame_idx = idx % num_frames;
    float mel_value = 0.0f;

    for (int i = 0; i < num_fft; ++i) {
        float input_value = __half2float(input[frame_idx * num_fft + i]);
        float filter_value = __half2float(mel_filters[mel_idx * num_fft + i]);
        mel_value += input_value * filter_value;
    }

    return static_cast<T>(logf(mel_value + 1.0f));
}

// Specialized version for __half
template <>
__device__ __half apply_mel_filter_and_log<__half, __half>(const __half* input, int idx, int num_features, int num_frames, const __half* mel_filters, int num_fft) {
    float mel_value = 0.0f;

    for (int i = 0; i < num_fft; ++i) {
        mel_value += __half2float(input[idx * num_fft + i]) * __half2float(mel_filters[idx * num_fft + i]);
    }

    return __float2half(logf(mel_value + 1.0f));
}

template <>
__device__ __nv_fp8_e4m3 apply_mel_filter_and_log<__nv_fp8_e4m3, __nv_fp8_e4m3>(
    const __nv_fp8_e4m3* input, int idx, int num_features, int num_frames, const __nv_fp8_e4m3* mel_filters, int num_fft) {

    float mel_value = 0.0f;

    // Loop through each frequency component in the FFT results for the given frame
    for (int i = 0; i < num_fft; ++i) {
        float input_value = fp8_to_float(input[idx * num_fft + i]);  // Convert fp8 to float for processing
        float filter_value = fp8_to_float(mel_filters[idx * num_fft + i]);  // Convert filter fp8 to float

        mel_value += input_value * filter_value;  // Multiply and accumulate the filtered input signal
    }

    // Apply logarithmic non-linearity and convert back to fp8
    return float_to_fp8(logf(mel_value + 1.0f));
}

// FP64 (double precision)
__global__ void mel_gram_fp64(const double* input, double* output, int num_features, int num_frames, const double* mel_filters, int num_fft) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features * num_frames) return;
    output[idx] = apply_mel_filter_and_log<double>(input, idx, num_features, num_frames, mel_filters, num_fft);
}

// FP32 (single precision)
__global__ void mel_gram_fp32(const float* input, float* output, int num_features, int num_frames, const float* mel_filters, int num_fft) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features * num_frames) return;
    output[idx] = apply_mel_filter_and_log<float>(input, idx, num_features, num_frames, mel_filters, num_fft);
}

// FP16 (half precision)
__global__ void mel_gram_fp16(const __half* input, __half* output, int num_features, int num_frames, const __half* mel_filters, int num_fft) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features * num_frames) return;
    output[idx] = apply_mel_filter_and_log<__half, __half>(input, idx, num_features, num_frames, mel_filters, num_fft);
}

// FP8 (nv_fp8_e4m3 precision) kernel
__global__ void mel_gram_fp8(const __nv_fp8_e4m3* input, __nv_fp8_e4m3* output, int num_features, int num_frames, const __nv_fp8_e4m3* mel_filters, int num_fft) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features * num_frames) return; // Guard against out-of-bounds access

    // Each thread computes one feature frame result using the apply_mel_filter_and_log function
    output[idx] = apply_mel_filter_and_log<__nv_fp8_e4m3, __nv_fp8_e4m3>(input, idx, num_features, num_frames, mel_filters, num_fft);
}


#endif // MEL_GRAM_KERNELS_CUH
