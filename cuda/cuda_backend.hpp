#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_fp16.h>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <vector>

namespace tinytorch {
namespace cuda {

// CUDA error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("CUBLAS error: " + std::to_string(status)); \
        } \
    } while(0)

#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            throw std::runtime_error("CUDNN error: " + std::string(cudnnGetErrorString(status))); \
        } \
    } while(0)

// CUDA device management
class CudaDevice {
public:
    static CudaDevice& get_instance() {
        static CudaDevice instance;
        return instance;
    }
    
    void set_device(int device_id) {
        CUDA_CHECK(cudaSetDevice(device_id));
        device_id_ = device_id;
    }
    
    int get_device() const { return device_id_; }
    
    cublasHandle_t get_cublas_handle() const { return cublas_handle_; }
    cudnnHandle_t get_cudnn_handle() const { return cudnn_handle_; }
    
    // Memory management
    void* allocate(size_t size) {
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        return ptr;
    }
    
    void deallocate(void* ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
    
    void copy_to_device(void* dst, const void* src, size_t size) {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    }
    
    void copy_to_host(void* dst, const void* src, size_t size) {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    }
    
    void copy_device_to_device(void* dst, const void* src, size_t size) {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    }
    
    void synchronize() {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    void get_device_properties() {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
        
        std::cout << "CUDA Device " << device_id_ << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Warp size: " << prop.warpSize << std::endl;
    }

private:
    CudaDevice() : device_id_(0) {
        // Initialize CUDA
        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            throw std::runtime_error("No CUDA devices found");
        }
        
        CUDA_CHECK(cudaSetDevice(device_id_));
        
        // Initialize cuBLAS
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        
        // Initialize cuDNN
        CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
        
        std::cout << "CUDA initialized with " << device_count << " devices" << std::endl;
    }
    
    ~CudaDevice() {
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
        }
        if (cudnn_handle_) {
            cudnnDestroy(cudnn_handle_);
        }
    }
    
    int device_id_;
    cublasHandle_t cublas_handle_;
    cudnnHandle_t cudnn_handle_;
};

// CUDA memory allocator
class CudaAllocator {
public:
    static void* allocate(size_t size) {
        return CudaDevice::get_instance().allocate(size);
    }
    
    static void deallocate(void* ptr) {
        CudaDevice::get_instance().deallocate(ptr);
    }
};

// CUDA tensor operations
namespace kernels {

// CUDA kernel declarations
extern "C" {
    // Basic operations
    void cuda_add_f32(float* dst, const float* src1, const float* src2, int n);
    void cuda_mul_f32(float* dst, const float* src1, const float* src2, int n);
    void cuda_add_f16(__half* dst, const __half* src1, const __half* src2, int n);
    void cuda_mul_f16(__half* dst, const __half* src1, const __half* src2, int n);
    
    // Matrix operations
    void cuda_matmul_f32(float* dst, const float* src1, const float* src2, 
                        int m, int n, int k, int batch_size);
    void cuda_matmul_f16(__half* dst, const __half* src1, const __half* src2, 
                        int m, int n, int k, int batch_size);
    
    // Activation functions
    void cuda_gelu_f32(float* dst, const float* src, int n);
    void cuda_gelu_f16(__half* dst, const __half* src, int n);
    void cuda_softmax_f32(float* dst, const float* src, int batch_size, int n);
    void cuda_softmax_f16(__half* dst, const __half* src, int batch_size, int n);
    
    // Normalization
    void cuda_layer_norm_f32(float* dst, const float* src, const float* weight, 
                            const float* bias, int batch_size, int n, float eps);
    void cuda_layer_norm_f16(__half* dst, const __half* src, const __half* weight, 
                            const __half* bias, int batch_size, int n, float eps);
    
    // Type conversions
    void cuda_f32_to_f16(__half* dst, const float* src, int n);
    void cuda_f16_to_f32(float* dst, const __half* src, int n);
}

// High-level operation wrappers
class CudaOps {
public:
    static void add(float* dst, const float* src1, const float* src2, int n) {
        cuda_add_f32(dst, src1, src2, n);
    }
    
    static void add(__half* dst, const __half* src1, const __half* src2, int n) {
        cuda_add_f16(dst, src1, src2, n);
    }
    
    static void mul(float* dst, const float* src1, const float* src2, int n) {
        cuda_mul_f32(dst, src1, src2, n);
    }
    
    static void mul(__half* dst, const __half* src1, const __half* src2, int n) {
        cuda_mul_f16(dst, src1, src2, n);
    }
    
    static void matmul(float* dst, const float* src1, const float* src2, 
                      int m, int n, int k, int batch_size = 1) {
        auto handle = CudaDevice::get_instance().get_cublas_handle();
        
        const float alpha = 1.0f, beta = 0.0f;
        
        if (batch_size == 1) {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   n, m, k, &alpha, src2, n, src1, k, &beta, dst, n));
        } else {
            CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k, &alpha, src2, n, n*k,
                                                 src1, k, m*k, &beta, dst, n, m*n,
                                                 batch_size));
        }
    }
    
    static void matmul(__half* dst, const __half* src1, const __half* src2, 
                      int m, int n, int k, int batch_size = 1) {
        auto handle = CudaDevice::get_instance().get_cublas_handle();
        
        const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        
        if (batch_size == 1) {
            CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   n, m, k, &alpha, src2, n, src1, k, &beta, dst, n));
        } else {
            CUBLAS_CHECK(cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, m, k, &alpha, src2, n, n*k,
                                                 src1, k, m*k, &beta, dst, n, m*n,
                                                 batch_size));
        }
    }
    
    static void gelu(float* dst, const float* src, int n) {
        cuda_gelu_f32(dst, src, n);
    }
    
    static void gelu(__half* dst, const __half* src, int n) {
        cuda_gelu_f16(dst, src, n);
    }
    
    static void softmax(float* dst, const float* src, int batch_size, int n) {
        cuda_softmax_f32(dst, src, batch_size, n);
    }
    
    static void softmax(__half* dst, const __half* src, int batch_size, int n) {
        cuda_softmax_f16(dst, src, batch_size, n);
    }
    
    static void layer_norm(float* dst, const float* src, const float* weight, 
                          const float* bias, int batch_size, int n, float eps = 1e-5f) {
        cuda_layer_norm_f32(dst, src, weight, bias, batch_size, n, eps);
    }
    
    static void layer_norm(__half* dst, const __half* src, const __half* weight, 
                          const __half* bias, int batch_size, int n, float eps = 1e-5f) {
        cuda_layer_norm_f16(dst, src, weight, bias, batch_size, n, eps);
    }
    
    static void convert_f32_to_f16(__half* dst, const float* src, int n) {
        cuda_f32_to_f16(dst, src, n);
    }
    
    static void convert_f16_to_f32(float* dst, const __half* src, int n) {
        cuda_f16_to_f32(dst, src, n);
    }
};

} // namespace kernels
} // namespace cuda
} // namespace tinytorch