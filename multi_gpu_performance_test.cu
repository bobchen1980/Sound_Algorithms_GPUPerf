#include <cfloat> 
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <cuda_runtime.h>
#include <nvml.h>
#include <chrono>
#include <nvml.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <chrono>
#include <iomanip>
#include <numeric> 
#include <thread>
#include <getopt.h>
#include "sound/matrix_mul_kernels.cuh"
#include "sound/octave_onethird_kernels.cuh"
#include "sound/mel_gram_kernels.cuh"

#define DEFAULT_TEST_ITERATIONS 10

using namespace std;
using namespace std::chrono;

#define CUDA_CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error in " << __FUNCTION__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << endl; \
            exit(1); \
        } \
    }

#define NVML_CHECK(call) \
    { \
        nvmlReturn_t err = call; \
        if (err != NVML_SUCCESS) { \
            cerr << "NVML error in " << __FUNCTION__ << " at line " << __LINE__ << ": " << nvmlErrorString(err) << endl; \
            exit(1); \
        } \
    }

struct GPUResult {
    int gpu_id;
    float avg_time, max_time, min_time;
    int iterations;
    std::string kernel_name;
    unsigned int temperature;          // GPU temperature in Celsius
    unsigned int gpu_utilization;      // GPU utilization percentage
    unsigned long long memory_used;    // GPU memory used in bytes
    unsigned long long total_memory;   // GPU memory utilization percentage
};

// Host function for float to fp8 conversion
__nv_fp8_e4m3 host_float_to_fp8(float x) {
    __half temp = __float2half(x);
    return *reinterpret_cast<__nv_fp8_e4m3*>(&temp);
}

std::vector<GPUResult> results;
std::mutex results_mutex;
std::mutex log_mutex;  // Global mutex for logging

// Function to handle logging and console output
void log_and_print(std::ofstream &log_file, std::ostringstream &message) {
    std::lock_guard<std::mutex> guard(log_mutex); // Locking for thread safety
    log_file << message.str() << std::endl;  // Log to file
    std::cout << message.str() << std::endl; // Print to console
    message.str("");
    message.clear();
}

void run_matmul_on_gpu(int gpu_id, std::ofstream &log_file, int test_iterations);

void run_octave_on_gpu(int gpu_id, std::ofstream& log_file, int test_iterations);

void run_mel_on_gpu(int gpu_id, std::ofstream& log_file, int test_iterations);

void logCudaLibraryVersions(ofstream &log_file);

void getGpuMetrics(int gpu_id, unsigned int &temperature, unsigned int &gpu_utilization, unsigned long long &memory_used, unsigned long long &total_memory);

int main(int argc, char *argv[]) {
    nvmlInit();

    int test_iterations = DEFAULT_TEST_ITERATIONS;

    // Parsing command line arguments
    int opt;
    static struct option long_options[] = {
        {"iter", required_argument, 0, 'i'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "i:", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'i':
                test_iterations = atoi(optarg);
                break;
            default:
                cerr << "Usage: " << argv[0] << " [--iter <iterations>]" << endl;
                exit(EXIT_FAILURE);
        }
    }    

    ofstream log_file("logs/gpu_performance_log.txt");
    vector<thread> gpu_threads;

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    std::ostringstream summary;
    summary << "Starting GPU Performance Test with " << device_count << " GPUs." << endl;
    log_and_print(log_file, summary);

    // for (int i = 0; i < device_count; ++i) {
    //     gpu_threads.emplace_back(run_mel_on_gpu, i, ref(log_file),test_iterations);
    // }
    // for (auto &t : gpu_threads) {
    //     t.join();
    // }
    // gpu_threads.clear();
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));  

    for (int i = 0; i < device_count; ++i) {
        gpu_threads.emplace_back(run_octave_on_gpu, i, ref(log_file),test_iterations);
    }
    for (auto &t : gpu_threads) {
        t.join();
    }
    gpu_threads.clear();    
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));  


    // for (int i = 0; i < device_count; ++i) {
    //     gpu_threads.emplace_back(run_matmul_on_gpu, i, std::ref(log_file),test_iterations);
    // }
    // for (auto& t : gpu_threads) {
    //     t.join();
    // }    
    // gpu_threads.clear();

    logCudaLibraryVersions(log_file);

    // Print results for each GPU including the new metrics
    for (const auto& result : results) {
        std::cout << "GPU " << result.gpu_id << " Summary: "  << result.kernel_name << " Avg Time = " << result.avg_time << "s, Min Time = "
                  << result.min_time << "s, Max Time = " << result.max_time << "s, Iterations = " << result.iterations
                  << ", Temperature = " << result.temperature << "C, GPU Utilization = " << result.gpu_utilization
                  << "%, Memory Used = " << result.memory_used / (1024 * 1024) << " MB, Total Memory = "
                  << result.total_memory / (1024 * 1024) << " MB" << std::endl;
    }

    nvmlShutdown();
    log_file.close();

    summary << "GPU performance testing completed." << endl;
    log_and_print(log_file, summary);
    return 0;
}

void logCudaLibraryVersions(ofstream &log_file) {
    int runtimeVersion = 0, driverVersion = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));

    std::ostringstream summary;
    summary << "CUDA Runtime Version: " << runtimeVersion << ", Driver Version: " << driverVersion << endl;

    // Check and log cuBLAS version
    int cublasVersion = 0;
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle); // Create a cuBLAS handle
    cublasGetVersion(cublasHandle, &cublasVersion);
    cublasDestroy(cublasHandle); // Destroy the handle to clean up
    summary << "cuBLAS Version: " << cublasVersion << endl;

    // Check and log cuFFT version
    int cufftVersion = 0;
    cufftGetVersion(&cufftVersion);
    summary << "cuFFT Version: " << cufftVersion << endl;
    log_and_print(log_file, summary);
    // Add any additional relevant library versions here
}

// Function to fetch GPU metrics
void getGpuMetrics(int gpu_id, unsigned int &temperature, unsigned int &gpu_utilization, unsigned long long &memory_used, unsigned long long &total_memory) {
    nvmlDevice_t device;
    NVML_CHECK(nvmlDeviceGetHandleByIndex(gpu_id, &device));

    // Get GPU temperature
    NVML_CHECK(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature));

    // Get GPU utilization and memory utilization
    nvmlUtilization_t utilization;
    NVML_CHECK(nvmlDeviceGetUtilizationRates(device, &utilization));
    gpu_utilization = utilization.gpu;

    // Get memory information
    nvmlMemory_t memory;
    NVML_CHECK(nvmlDeviceGetMemoryInfo(device, &memory));
    memory_used = memory.used;
    total_memory = memory.total;
}


void run_matmul_on_gpu(int gpu_id, std::ofstream &log_file, int test_iterations) {
    CUDA_CHECK(cudaSetDevice(gpu_id));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int N = 4096;
    size_t bytes_fp64 = N * N * sizeof(double);
    size_t bytes_fp32 = N * N * sizeof(float);
    size_t bytes_fp16 = N * N * sizeof(__half);
    size_t bytes_fp8 = N * N * sizeof(__nv_fp8_e4m3);  // Assuming __nv_fp8_e4m3 is defined somewhere

    // Allocate memory for all data types
    double *d_A64, *d_B64, *d_C64;
    float *d_A32, *d_B32, *d_C32;
    __half *d_A16, *d_B16, *d_C16;
    __nv_fp8_e4m3 *d_A8, *d_B8, *d_C8;  // For the 8-bit data type

    CUDA_CHECK(cudaMalloc(&d_A64, bytes_fp64));
    CUDA_CHECK(cudaMalloc(&d_B64, bytes_fp64));
    CUDA_CHECK(cudaMalloc(&d_C64, bytes_fp64));

    CUDA_CHECK(cudaMalloc(&d_A32, bytes_fp32));
    CUDA_CHECK(cudaMalloc(&d_B32, bytes_fp32));
    CUDA_CHECK(cudaMalloc(&d_C32, bytes_fp32));

    CUDA_CHECK(cudaMalloc(&d_A16, bytes_fp16));
    CUDA_CHECK(cudaMalloc(&d_B16, bytes_fp16));
    CUDA_CHECK(cudaMalloc(&d_C16, bytes_fp16));

    CUDA_CHECK(cudaMalloc(&d_A8, bytes_fp8));
    CUDA_CHECK(cudaMalloc(&d_B8, bytes_fp8));
    CUDA_CHECK(cudaMalloc(&d_C8, bytes_fp8));

    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    dim3 threads(16, 16);

    // Define the names of the kernels for logging purposes
    const char* kernel_names[] = {"matmul_fp64", "matmul_fp32", "matmul_fp16", "matmul_fp8"};

    // Fetch GPU metrics before the loop of kernel execution
    unsigned int temperature, gpu_utilization;
    unsigned long long memory_used, total_memory;
    getGpuMetrics(gpu_id, temperature, gpu_utilization, memory_used, total_memory);

    // matmul_fp64<<<blocks, threads, 0, stream>>>(d_A64, d_B64, d_C64, N);
    // CUDA_CHECK(cudaStreamSynchronize(stream));  // 确保预热内核完成

    // Loop to run each set of tests
    for (int type = 0; type < 4; type++) {
        std::vector<float> execution_times;
        auto start_total = std::chrono::high_resolution_clock::now();
        float duration = 0, max_time = 0, min_time = FLT_MAX;
        int iterations = 0;

        auto start_avg = std::chrono::high_resolution_clock::now();
        while ( iterations < test_iterations * 100 ) {
            auto start = std::chrono::high_resolution_clock::now();

            // Execute the appropriate kernel
            switch (type) {
                case 0:
                    matmul_fp64<<<blocks, threads>>>(d_A64, d_B64, d_C64, N);
                    break;
                case 1:
                    matmul_fp32<<<blocks, threads>>>(d_A32, d_B32, d_C32, N);
                    break;
                case 2:
                    matmul_fp16<<<blocks, threads>>>(d_A16, d_B16, d_C16, N);
                    break;
                case 3:
                    matmul_fp8<<<blocks, threads>>>(d_A8, d_B8, d_C8, N);
                    break;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));  // 确保内核执行完毕

            auto end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f;
            if( duration > 0 ) {
                execution_times.push_back(duration);
            }       
            // execution_times.push_back(duration);    
            max_time = std::max(max_time, duration);
            min_time = std::min(min_time, duration);

            iterations++;

            std::ostringstream ss;
            ss << "GPU " << gpu_id << " Func " << std::string(kernel_names[type]) << ": Exec Time = " << duration << "s, Iterations = " << iterations;

            log_and_print(log_file, ss);

        }

        auto end_avg = std::chrono::high_resolution_clock::now();
        // float avg_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0f) / execution_times.size();
        float avg_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_avg - start_avg).count() / iterations / 1000.0f;
        std::lock_guard<std::mutex> guard(results_mutex);
        results.push_back(GPUResult{gpu_id, avg_time, max_time, min_time, iterations, std::string(kernel_names[type]),
                                    temperature, gpu_utilization, memory_used, total_memory});

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));                            
    }

    // Free memory and destroy stream
    cudaFree(d_A64); cudaFree(d_B64); cudaFree(d_C64);
    cudaFree(d_A32); cudaFree(d_B32); cudaFree(d_C32);
    cudaFree(d_A16); cudaFree(d_B16); cudaFree(d_C16);
    cudaFree(d_A8); cudaFree(d_B8); cudaFree(d_C8);

    cudaStreamDestroy(stream);
}

void run_octave_on_gpu(int gpu_id, std::ofstream& log_file, int test_iterations) {
    CUDA_CHECK(cudaSetDevice(gpu_id));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int N = 40960;
    const int num_bands = 10; // 根据实际需要设置
    const double f_base = 1000.0; // 基础频率，按需设置
    const double k_base_value = pow(2.0, 1.0 / 3.0);

    // Copy the k_base_value to the constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(k_base, &k_base_value, sizeof(double)));

    size_t bytes_fp64 = N * sizeof(double);
    size_t bytes_fp32 = N * sizeof(float);
    size_t bytes_fp16 = N * sizeof(__half);
    size_t bytes_fp8 = N * sizeof(__nv_fp8_e4m3); // Assuming __nv_fp8_e4m3 is defined somewhere

    // Allocate memory for all data types
    double *d_A64, *d_B64, *d_C64;
    float *d_A32, *d_B32, *d_C32;
    __half *d_A16, *d_B16, *d_C16;
    __nv_fp8_e4m3 *d_A8, *d_B8, *d_C8; // For the 8-bit data type

    CUDA_CHECK(cudaMalloc(&d_A64, bytes_fp64));
    CUDA_CHECK(cudaMalloc(&d_B64, bytes_fp64));
    CUDA_CHECK(cudaMalloc(&d_C64, num_bands * sizeof(double))); // Output size depends on num_bands

    CUDA_CHECK(cudaMalloc(&d_A32, bytes_fp32));
    CUDA_CHECK(cudaMalloc(&d_B32, bytes_fp32));
    CUDA_CHECK(cudaMalloc(&d_C32, num_bands * sizeof(float))); // Output size depends on num_bands

    CUDA_CHECK(cudaMalloc(&d_A16, bytes_fp16));
    CUDA_CHECK(cudaMalloc(&d_B16, bytes_fp16));
    CUDA_CHECK(cudaMalloc(&d_C16, num_bands * sizeof(__half))); // Output size depends on num_bands

    CUDA_CHECK(cudaMalloc(&d_A8, bytes_fp8));
    CUDA_CHECK(cudaMalloc(&d_B8, bytes_fp8));
    CUDA_CHECK(cudaMalloc(&d_C8, num_bands * sizeof(__nv_fp8_e4m3))); // Output size depends on num_bands

    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    dim3 threads(16, 16);

    // Define the names of the kernels for logging purposes
    const char* kernel_names[] = {"octave_fp64", "octave_fp32", "octave_fp16", "octave_fp8"};

    // Fetch GPU metrics before the loop of kernel execution
    unsigned int temperature, gpu_utilization;
    unsigned long long memory_used, total_memory;
    getGpuMetrics(gpu_id, temperature, gpu_utilization, memory_used, total_memory);

    // Loop to run each set of tests
    for (int type = 0; type < 4; type++) {
        std::vector<float> execution_times;
        auto start_total = std::chrono::high_resolution_clock::now();
        float duration = 0, max_time = 0, min_time = FLT_MAX;
        int iterations = 0;

        auto start_avg = std::chrono::high_resolution_clock::now();
        while ( iterations < test_iterations ) {
            auto start = std::chrono::high_resolution_clock::now();

            // Execute the appropriate kernel
            switch (type) {
                case 0:
                    octave_kernel_fp64<<<blocks, threads, 0, stream>>>(d_A64, d_C64, N, num_bands, f_base);
                    break;
                case 1:
                    octave_kernel_fp32<<<blocks, threads, 0, stream>>>(d_A32, d_C32, N, num_bands, f_base);
                    break;
                case 2:
                    octave_kernel_fp16<<<blocks, threads, 0, stream>>>(d_A16, d_C16, N, num_bands, __float2half(f_base));
                    break;
                case 3:
                    octave_kernel_fp8<<<blocks, threads, 0, stream>>>(d_A8, d_C8, N, num_bands, __float2half(f_base));
                    break;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));  // Ensure kernel execution is complete

            auto end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f;
            if (duration > 0) {
                execution_times.push_back(duration);
            }
            max_time = std::max(max_time, duration);
            min_time = std::min(min_time, duration);

            iterations++;

            std::ostringstream ss;
            ss << "GPU " << gpu_id << " Func " << std::string(kernel_names[type]) << ": Exec Time = " << duration << "s, Iterations = " << iterations;

            log_and_print(log_file, ss);

        }

        auto end_avg = std::chrono::high_resolution_clock::now();
        // float avg_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0f) / execution_times.size();
        float avg_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_avg - start_avg).count() / iterations / 1000.0f;
        std::lock_guard<std::mutex> guard(results_mutex);
        results.push_back(GPUResult{gpu_id, avg_time, max_time, min_time, iterations, std::string(kernel_names[type]),
                                    temperature, gpu_utilization, memory_used, total_memory});
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));                            
    }

    // Free memory and destroy stream
    cudaFree(d_A64); cudaFree(d_B64); cudaFree(d_C64);
    cudaFree(d_A32); cudaFree(d_B32); cudaFree(d_C32);
    cudaFree(d_A16); cudaFree(d_B16); cudaFree(d_C16);
    cudaFree(d_A8); cudaFree(d_B8); cudaFree(d_C8);

    cudaStreamDestroy(stream);
}

void run_mel_on_gpu(int gpu_id, std::ofstream& log_file, int test_iterations) {
    CUDA_CHECK(cudaSetDevice(gpu_id));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int N = 40960;
    const int num_fft = 1024; // Example FFT size, adjust as necessary
    const int sample_rate = 16000; // Example sample rate, adjust as necessary
    const int num_mel_filters = NUM_MEL_FILTERS;

    size_t bytes_fp64 = N * sizeof(double);
    size_t bytes_fp32 = N * sizeof(float);
    size_t bytes_fp16 = N * sizeof(__half);
    size_t bytes_fp8 = N * sizeof(__nv_fp8_e4m3); // Assuming __nv_fp8_e4m3 is defined somewhere

    size_t mel_filter_bytes_fp64 = num_mel_filters * (num_fft / 2 + 1) * sizeof(double);
    size_t mel_filter_bytes_fp32 = num_mel_filters * (num_fft / 2 + 1) * sizeof(float);
    size_t mel_filter_bytes_fp16 = num_mel_filters * (num_fft / 2 + 1) * sizeof(__half);
    size_t mel_filter_bytes_fp8 = num_mel_filters * (num_fft / 2 + 1) * sizeof(__nv_fp8_e4m3);

    // Allocate memory for all data types
    double *d_A64, *d_C64, *d_mel_filters64;
    float *d_A32, *d_C32, *d_mel_filters32;
    __half *d_A16, *d_C16, *d_mel_filters16;
    __nv_fp8_e4m3 *d_A8, *d_C8, *d_mel_filters8; // For the 8-bit data type

    CUDA_CHECK(cudaMalloc(&d_A64, bytes_fp64));
    CUDA_CHECK(cudaMalloc(&d_C64, num_mel_filters * sizeof(double))); // Output size depends on num_mel_filters
    CUDA_CHECK(cudaMalloc(&d_mel_filters64, mel_filter_bytes_fp64));

    CUDA_CHECK(cudaMalloc(&d_A32, bytes_fp32));
    CUDA_CHECK(cudaMalloc(&d_C32, num_mel_filters * sizeof(float))); // Output size depends on num_mel_filters
    CUDA_CHECK(cudaMalloc(&d_mel_filters32, mel_filter_bytes_fp32));

    CUDA_CHECK(cudaMalloc(&d_A16, bytes_fp16));
    CUDA_CHECK(cudaMalloc(&d_C16, num_mel_filters * sizeof(__half))); // Output size depends on num_mel_filters
    CUDA_CHECK(cudaMalloc(&d_mel_filters16, mel_filter_bytes_fp16));

    CUDA_CHECK(cudaMalloc(&d_A8, bytes_fp8));
    CUDA_CHECK(cudaMalloc(&d_C8, num_mel_filters * sizeof(__nv_fp8_e4m3))); // Output size depends on num_mel_filters
    CUDA_CHECK(cudaMalloc(&d_mel_filters8, mel_filter_bytes_fp8));

    // Initialize input and filter memory with zeros
    CUDA_CHECK(cudaMemset(d_A64, 0, bytes_fp64));
    CUDA_CHECK(cudaMemset(d_C64, 0, num_mel_filters * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_mel_filters64, 0, mel_filter_bytes_fp64));

    CUDA_CHECK(cudaMemset(d_A32, 0, bytes_fp32));
    CUDA_CHECK(cudaMemset(d_C32, 0, num_mel_filters * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mel_filters32, 0, mel_filter_bytes_fp32));

    CUDA_CHECK(cudaMemset(d_A16, 0, bytes_fp16));
    CUDA_CHECK(cudaMemset(d_C16, 0, num_mel_filters * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_mel_filters16, 0, mel_filter_bytes_fp16));

    CUDA_CHECK(cudaMemset(d_A8, 0, bytes_fp8));
    CUDA_CHECK(cudaMemset(d_C8, 0, num_mel_filters * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMemset(d_mel_filters8, 0, mel_filter_bytes_fp8));

    // Generate mel filter banks
    float* h_mel_filters = (float*)malloc(mel_filter_bytes_fp32);
    generate_mel_filter_bank(h_mel_filters, num_mel_filters, num_fft, sample_rate);

    CUDA_CHECK(cudaMemcpy(d_mel_filters64, h_mel_filters, mel_filter_bytes_fp64, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mel_filters32, h_mel_filters, mel_filter_bytes_fp32, cudaMemcpyHostToDevice));
    // For half precision, we need to convert
    __half* h_mel_filters16 = (__half*)malloc(mel_filter_bytes_fp16);
    for (int i = 0; i < num_mel_filters * (num_fft / 2 + 1); ++i) {
        h_mel_filters16[i] = __float2half(h_mel_filters[i]);
    }
    CUDA_CHECK(cudaMemcpy(d_mel_filters16, h_mel_filters16, mel_filter_bytes_fp16, cudaMemcpyHostToDevice));
    // For FP8, we need to convert
    __nv_fp8_e4m3* h_mel_filters8 = (__nv_fp8_e4m3*)malloc(mel_filter_bytes_fp8);
    for (int i = 0; i < num_mel_filters * (num_fft / 2 + 1); ++i) {
        h_mel_filters8[i] = host_float_to_fp8(h_mel_filters[i]);
    }
    CUDA_CHECK(cudaMemcpy(d_mel_filters8, h_mel_filters8, mel_filter_bytes_fp8, cudaMemcpyHostToDevice));

    free(h_mel_filters);
    free(h_mel_filters16);
    free(h_mel_filters8);

    dim3 blocks(16, 16);
    dim3 threads(16, 16);

    // Define the names of the kernels for logging purposes
    const char* kernel_names[] = {"mel_gram_fp64", "mel_gram_fp32", "mel_gram_fp16", "mel_gram_fp8"};

    // Fetch GPU metrics before the loop of kernel execution
    unsigned int temperature, gpu_utilization;
    unsigned long long memory_used, total_memory;
    getGpuMetrics(gpu_id, temperature, gpu_utilization, memory_used, total_memory);

    // Loop to run each set of tests
    for (int type = 0; type < 4; type++) {
        std::vector<float> execution_times;
        auto start_total = std::chrono::high_resolution_clock::now();
        float duration = 0, max_time = 0, min_time = FLT_MAX;
        int iterations = 0;

        auto start_avg = std::chrono::high_resolution_clock::now();
        while (iterations < test_iterations) {
            auto start = std::chrono::high_resolution_clock::now();

            // Execute the appropriate kernel
            switch (type) {
                case 0:
                    mel_gram_fp64<<<blocks, threads, 0, stream>>>(d_A64, d_C64, num_mel_filters, N, d_mel_filters64, num_fft);
                    break;
                case 1:
                    mel_gram_fp32<<<blocks, threads, 0, stream>>>(d_A32, d_C32, num_mel_filters, N, d_mel_filters32, num_fft);
                    break;
                case 2:
                    mel_gram_fp16<<<blocks, threads, 0, stream>>>(d_A16, d_C16, num_mel_filters, N, d_mel_filters16, num_fft);
                    break;
                case 3:
                    mel_gram_fp8<<<blocks, threads, 0, stream>>>(d_A8, d_C8, num_mel_filters, N, d_mel_filters8, num_fft);
                    break;
            }

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));  // Ensure kernel execution is complete

            auto end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f;
            if (duration > 0) {
                execution_times.push_back(duration);
            }
            max_time = std::max(max_time, duration);
            min_time = std::min(min_time, duration);

            iterations++;

            std::ostringstream ss;
            ss << "GPU " << gpu_id << " Func " << std::string(kernel_names[type]) << ": Exec Time = " << duration << "s, Iterations = " << iterations;

            log_and_print(log_file, ss);
        }

        auto end_avg = std::chrono::high_resolution_clock::now();
        float avg_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_avg - start_avg).count() / iterations / 1000.0f;
        std::lock_guard<std::mutex> guard(results_mutex);
        results.push_back(GPUResult{gpu_id, avg_time, max_time, min_time, iterations, std::string(kernel_names[type]),
                                    temperature, gpu_utilization, memory_used, total_memory});
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));                            
    }

    // Free memory and destroy stream
    CUDA_CHECK(cudaFree(d_A64)); CUDA_CHECK(cudaFree(d_C64)); CUDA_CHECK(cudaFree(d_mel_filters64));
    CUDA_CHECK(cudaFree(d_A32)); CUDA_CHECK(cudaFree(d_C32)); CUDA_CHECK(cudaFree(d_mel_filters32));
    CUDA_CHECK(cudaFree(d_A16)); CUDA_CHECK(cudaFree(d_C16)); CUDA_CHECK(cudaFree(d_mel_filters16));
    CUDA_CHECK(cudaFree(d_A8)); CUDA_CHECK(cudaFree(d_C8)); CUDA_CHECK(cudaFree(d_mel_filters8));

    CUDA_CHECK(cudaStreamDestroy(stream));
}
