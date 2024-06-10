# Makefile

CUDA_PATH := /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
NVML_LIB := -lnvidia-ml
CUBLAS_LIB := -lcublas
CUFFT_LIB := -lcufft

INCLUDES := -I$(CUDA_PATH)/include -Isound

TARGET := multi_gpu_performance_test
SRCS := multi_gpu_performance_test.cu
OBJS := $(SRCS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) -o $@ $^ $(CUBLAS_LIB) $(CUFFT_LIB) $(NVML_LIB)

%.o: %.cu
	$(NVCC) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: clean
