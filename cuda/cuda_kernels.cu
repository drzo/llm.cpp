#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Thread block size
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

// Utility functions
__device__ float gelu_f32(float x) {
    return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
}

__device__ __half gelu_f16(__half x) {
    float xf = __half2float(x);
    return __float2half(gelu_f32(xf));
}

// Basic arithmetic operations
__global__ void add_f32_kernel(float* dst, const float* src1, const float* src2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src1[idx] + src2[idx];
    }
}

__global__ void add_f16_kernel(__half* dst, const __half* src1, const __half* src2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __hadd(src1[idx], src2[idx]);
    }
}

__global__ void mul_f32_kernel(float* dst, const float* src1, const float* src2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src1[idx] * src2[idx];
    }
}

__global__ void mul_f16_kernel(__half* dst, const __half* src1, const __half* src2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __hmul(src1[idx], src2[idx]);
    }
}

// Activation functions
__global__ void gelu_f32_kernel(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = gelu_f32(src[idx]);
    }
}

__global__ void gelu_f16_kernel(__half* dst, const __half* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = gelu_f16(src[idx]);
    }
}

// Softmax implementation
__global__ void softmax_f32_kernel(float* dst, const float* src, int batch_size, int n) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* input = src + batch_idx * n;
    float* output = dst + batch_idx * n;
    
    // Find max value for numerical stability
    __shared__ float max_val;
    if (threadIdx.x == 0) {
        max_val = input[0];
        for (int i = 1; i < n; i++) {
            max_val = fmaxf(max_val, input[i]);
        }
    }
    __syncthreads();
    
    // Compute exp and sum
    __shared__ float sum_val;
    if (threadIdx.x == 0) {
        sum_val = 0.0f;
        for (int i = 0; i < n; i++) {
            float exp_val = expf(input[i] - max_val);
            output[i] = exp_val;
            sum_val += exp_val;
        }
    }
    __syncthreads();
    
    // Normalize
    int idx = threadIdx.x;
    while (idx < n) {
        output[idx] /= sum_val;
        idx += blockDim.x;
    }
}

__global__ void softmax_f16_kernel(__half* dst, const __half* src, int batch_size, int n) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const __half* input = src + batch_idx * n;
    __half* output = dst + batch_idx * n;
    
    // Find max value for numerical stability
    __shared__ float max_val;
    if (threadIdx.x == 0) {
        max_val = __half2float(input[0]);
        for (int i = 1; i < n; i++) {
            max_val = fmaxf(max_val, __half2float(input[i]));
        }
    }
    __syncthreads();
    
    // Compute exp and sum
    __shared__ float sum_val;
    if (threadIdx.x == 0) {
        sum_val = 0.0f;
        for (int i = 0; i < n; i++) {
            float exp_val = expf(__half2float(input[i]) - max_val);
            output[i] = __float2half(exp_val);
            sum_val += exp_val;
        }
    }
    __syncthreads();
    
    // Normalize
    int idx = threadIdx.x;
    while (idx < n) {
        output[idx] = __hdiv(output[idx], __float2half(sum_val));
        idx += blockDim.x;
    }
}

// Layer normalization
__global__ void layer_norm_f32_kernel(float* dst, const float* src, const float* weight, 
                                     const float* bias, int batch_size, int n, float eps) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const float* input = src + batch_idx * n;
    float* output = dst + batch_idx * n;
    
    // Compute mean
    __shared__ float mean;
    if (threadIdx.x == 0) {
        mean = 0.0f;
        for (int i = 0; i < n; i++) {
            mean += input[i];
        }
        mean /= n;
    }
    __syncthreads();
    
    // Compute variance
    __shared__ float var;
    if (threadIdx.x == 0) {
        var = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = input[i] - mean;
            var += diff * diff;
        }
        var = var / n + eps;
    }
    __syncthreads();
    
    // Normalize
    float inv_std = rsqrtf(var);
    int idx = threadIdx.x;
    while (idx < n) {
        float normalized = (input[idx] - mean) * inv_std;
        output[idx] = normalized * weight[idx] + bias[idx];
        idx += blockDim.x;
    }
}

__global__ void layer_norm_f16_kernel(__half* dst, const __half* src, const __half* weight, 
                                     const __half* bias, int batch_size, int n, float eps) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    const __half* input = src + batch_idx * n;
    __half* output = dst + batch_idx * n;
    
    // Compute mean
    __shared__ float mean;
    if (threadIdx.x == 0) {
        mean = 0.0f;
        for (int i = 0; i < n; i++) {
            mean += __half2float(input[i]);
        }
        mean /= n;
    }
    __syncthreads();
    
    // Compute variance
    __shared__ float var;
    if (threadIdx.x == 0) {
        var = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = __half2float(input[i]) - mean;
            var += diff * diff;
        }
        var = var / n + eps;
    }
    __syncthreads();
    
    // Normalize
    float inv_std = rsqrtf(var);
    int idx = threadIdx.x;
    while (idx < n) {
        float normalized = (__half2float(input[idx]) - mean) * inv_std;
        float result = normalized * __half2float(weight[idx]) + __half2float(bias[idx]);
        output[idx] = __float2half(result);
        idx += blockDim.x;
    }
}

// Type conversion kernels
__global__ void f32_to_f16_kernel(__half* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void f16_to_f32_kernel(float* dst, const __half* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

// C interface functions
extern "C" {

void cuda_add_f32(float* dst, const float* src1, const float* src2, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_f32_kernel<<<grid_size, BLOCK_SIZE>>>(dst, src1, src2, n);
}

void cuda_add_f16(__half* dst, const __half* src1, const __half* src2, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_f16_kernel<<<grid_size, BLOCK_SIZE>>>(dst, src1, src2, n);
}

void cuda_mul_f32(float* dst, const float* src1, const float* src2, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mul_f32_kernel<<<grid_size, BLOCK_SIZE>>>(dst, src1, src2, n);
}

void cuda_mul_f16(__half* dst, const __half* src1, const __half* src2, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mul_f16_kernel<<<grid_size, BLOCK_SIZE>>>(dst, src1, src2, n);
}

void cuda_gelu_f32(float* dst, const float* src, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gelu_f32_kernel<<<grid_size, BLOCK_SIZE>>>(dst, src, n);
}

void cuda_gelu_f16(__half* dst, const __half* src, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gelu_f16_kernel<<<grid_size, BLOCK_SIZE>>>(dst, src, n);
}

void cuda_softmax_f32(float* dst, const float* src, int batch_size, int n) {
    softmax_f32_kernel<<<batch_size, BLOCK_SIZE>>>(dst, src, batch_size, n);
}

void cuda_softmax_f16(__half* dst, const __half* src, int batch_size, int n) {
    softmax_f16_kernel<<<batch_size, BLOCK_SIZE>>>(dst, src, batch_size, n);
}

void cuda_layer_norm_f32(float* dst, const float* src, const float* weight, 
                         const float* bias, int batch_size, int n, float eps) {
    layer_norm_f32_kernel<<<batch_size, BLOCK_SIZE>>>(dst, src, weight, bias, batch_size, n, eps);
}

void cuda_layer_norm_f16(__half* dst, const __half* src, const __half* weight, 
                         const __half* bias, int batch_size, int n, float eps) {
    layer_norm_f16_kernel<<<batch_size, BLOCK_SIZE>>>(dst, src, weight, bias, batch_size, n, eps);
}

void cuda_f32_to_f16(__half* dst, const float* src, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    f32_to_f16_kernel<<<grid_size, BLOCK_SIZE>>>(dst, src, n);
}

void cuda_f16_to_f32(float* dst, const __half* src, int n) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    f16_to_f32_kernel<<<grid_size, BLOCK_SIZE>>>(dst, src, n);
}

} // extern "C"