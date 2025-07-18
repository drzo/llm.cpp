#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include "../tinytorch.hpp"

#ifdef ENABLE_CUDA
#include "../cuda/cuda_backend.hpp"
#endif

namespace tinytorch {
namespace amp {

// Mixed precision configuration
struct AMPConfig {
    bool enabled = false;
    float loss_scale = 65536.0f;  // Initial loss scale
    float min_loss_scale = 1.0f;  // Minimum loss scale
    float max_loss_scale = 65536.0f;  // Maximum loss scale
    int growth_factor = 2;         // Scale growth factor
    int backoff_factor = 2;        // Scale backoff factor
    int growth_interval = 2000;    // Steps before growing scale
    int consecutive_failures = 0;  // Consecutive overflow failures
    int max_consecutive_failures = 5;  // Max failures before scaling down
    bool skip_update = false;      // Skip optimizer update on overflow
    
    // Operations to keep in FP32
    std::vector<TensorOp> fp32_ops = {
        kOpSoftmax, kOpCrossEntropy, kOpNorm
    };
    
    bool should_use_fp32(TensorOp op) const {
        return std::find(fp32_ops.begin(), fp32_ops.end(), op) != fp32_ops.end();
    }
};

// Loss scaler for mixed precision training
class LossScaler {
public:
    explicit LossScaler(const AMPConfig& config) : config_(config) {}
    
    float get_scale() const { return config_.loss_scale; }
    
    void scale_loss(Tensor* loss) {
        if (config_.enabled && config_.loss_scale > 1.0f) {
            auto scaled_loss = *loss * config_.loss_scale;
            loss->CopyDataFrom(scaled_loss);
        }
    }
    
    void unscale_gradients(const std::vector<Tensor*>& parameters) {
        if (!config_.enabled || config_.loss_scale <= 1.0f) return;
        
        float inv_scale = 1.0f / config_.loss_scale;
        
        for (auto* param : parameters) {
            if (param->grad()) {
                auto grad_data = param->grad()->Flatten();
                for (auto& val : grad_data) {
                    val *= inv_scale;
                }
                param->grad()->Fill(grad_data);
            }
        }
    }
    
    bool check_overflow(const std::vector<Tensor*>& parameters) {
        if (!config_.enabled) return false;
        
        for (auto* param : parameters) {
            if (param->grad()) {
                auto grad_data = param->grad()->Flatten();
                for (float val : grad_data) {
                    if (std::isnan(val) || std::isinf(val)) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    void update_scale(bool has_overflow) {
        if (!config_.enabled) return;
        
        if (has_overflow) {
            config_.consecutive_failures++;
            if (config_.consecutive_failures >= config_.max_consecutive_failures) {
                config_.loss_scale = std::max(config_.min_loss_scale, 
                                            config_.loss_scale / config_.backoff_factor);
                config_.consecutive_failures = 0;
                config_.skip_update = true;
                std::cout << "Loss scale reduced to: " << config_.loss_scale << std::endl;
            }
        } else {
            config_.consecutive_failures = 0;
            config_.skip_update = false;
            
            // Grow loss scale periodically
            static int steps_since_growth = 0;
            steps_since_growth++;
            if (steps_since_growth >= config_.growth_interval) {
                config_.loss_scale = std::min(config_.max_loss_scale, 
                                            config_.loss_scale * config_.growth_factor);
                steps_since_growth = 0;
                std::cout << "Loss scale increased to: " << config_.loss_scale << std::endl;
            }
        }
    }
    
    bool should_skip_update() const {
        return config_.skip_update;
    }

private:
    mutable AMPConfig config_;
};

// Mixed precision tensor manager
class MixedPrecisionManager {
public:
    explicit MixedPrecisionManager(const AMPConfig& config) 
        : config_(config), loss_scaler_(config) {}
    
    // Convert tensor to appropriate precision
    Tensor* to_compute_precision(Tensor* tensor) {
        if (!config_.enabled) return tensor;
        
        TensorOp op = tensor->op_;
        if (config_.should_use_fp32(op)) {
            return ensure_fp32(tensor);
        } else {
            return ensure_fp16(tensor);
        }
    }
    
    // Ensure tensor is in FP32
    Tensor* ensure_fp32(Tensor* tensor) {
        if (tensor->type() == kF32) return tensor;
        
        // Convert from FP16 to FP32
        auto* fp32_tensor = tensor->ctx_->NewTensor(tensor->Dims(), kF32);
        
#ifdef ENABLE_CUDA
        if (is_cuda_tensor(tensor)) {
            cuda::kernels::CudaOps::convert_f16_to_f32(
                (float*)fp32_tensor->data(),
                (__half*)tensor->data(),
                tensor->NumElements()
            );
        } else {
#endif
            // CPU conversion
            auto fp16_data = tensor->Flatten<uint16_t>();
            std::vector<float> fp32_data(fp16_data.size());
            for (size_t i = 0; i < fp16_data.size(); i++) {
                fp32_data[i] = half_to_float(fp16_data[i]);
            }
            fp32_tensor->Fill(fp32_data);
#ifdef ENABLE_CUDA
        }
#endif
        
        return fp32_tensor;
    }
    
    // Ensure tensor is in FP16
    Tensor* ensure_fp16(Tensor* tensor) {
        if (tensor->type() == kF16) return tensor;
        
        // Convert from FP32 to FP16
        auto* fp16_tensor = tensor->ctx_->NewTensor(tensor->Dims(), kF16);
        
#ifdef ENABLE_CUDA
        if (is_cuda_tensor(tensor)) {
            cuda::kernels::CudaOps::convert_f32_to_f16(
                (__half*)fp16_tensor->data(),
                (float*)tensor->data(),
                tensor->NumElements()
            );
        } else {
#endif
            // CPU conversion
            auto fp32_data = tensor->Flatten<float>();
            std::vector<uint16_t> fp16_data(fp32_data.size());
            for (size_t i = 0; i < fp32_data.size(); i++) {
                fp16_data[i] = float_to_half(fp32_data[i]);
            }
            fp16_tensor->Fill(fp16_data);
#ifdef ENABLE_CUDA
        }
#endif
        
        return fp16_tensor;
    }
    
    LossScaler& get_loss_scaler() { return loss_scaler_; }
    
    const AMPConfig& get_config() const { return config_; }

private:
    AMPConfig config_;
    LossScaler loss_scaler_;
    
    // Helper functions for FP16 conversion
    static float half_to_float(uint16_t half_val) {
        // Simple IEEE 754 binary16 to binary32 conversion
        uint32_t sign = (half_val & 0x8000) << 16;
        uint32_t exp = (half_val & 0x7C00) >> 10;
        uint32_t mant = half_val & 0x03FF;
        
        if (exp == 0) {
            if (mant == 0) {
                return *reinterpret_cast<float*>(&sign);
            } else {
                // Denormalized number
                exp = 127 - 14;
                while (!(mant & 0x0400)) {
                    mant <<= 1;
                    exp--;
                }
                mant &= 0x03FF;
            }
        } else if (exp == 0x1F) {
            exp = 0xFF;
        } else {
            exp += 127 - 15;
        }
        
        uint32_t result = sign | (exp << 23) | (mant << 13);
        return *reinterpret_cast<float*>(&result);
    }
    
    static uint16_t float_to_half(float float_val) {
        // Simple IEEE 754 binary32 to binary16 conversion
        uint32_t bits = *reinterpret_cast<uint32_t*>(&float_val);
        uint16_t sign = (bits & 0x80000000) >> 16;
        uint32_t exp = (bits & 0x7F800000) >> 23;
        uint32_t mant = bits & 0x007FFFFF;
        
        if (exp == 0) {
            return sign;
        } else if (exp == 0xFF) {
            return sign | 0x7C00 | (mant ? 0x0200 : 0);
        } else {
            int32_t new_exp = exp - 127 + 15;
            if (new_exp >= 0x1F) {
                return sign | 0x7C00;
            } else if (new_exp <= 0) {
                return sign;
            } else {
                return sign | (new_exp << 10) | (mant >> 13);
            }
        }
    }
    
#ifdef ENABLE_CUDA
    static bool is_cuda_tensor(Tensor* tensor) {
        // Check if tensor is allocated on GPU
        cudaPointerAttributes attributes;
        cudaError_t error = cudaPointerGetAttributes(&attributes, tensor->data());
        return error == cudaSuccess && attributes.devicePointer != nullptr;
    }
#endif
};

// Automatic Mixed Precision trainer
class AMPTrainer {
public:
    AMPTrainer(const std::vector<Tensor*>& parameters, const AMPConfig& config)
        : parameters_(parameters), mp_manager_(config) {}
    
    void forward_backward(Tensor* loss) {
        // Scale loss for mixed precision
        mp_manager_.get_loss_scaler().scale_loss(loss);
        
        // Backward pass
        loss->Backward();
        
        // Unscale gradients
        mp_manager_.get_loss_scaler().unscale_gradients(parameters_);
        
        // Check for overflow
        bool has_overflow = mp_manager_.get_loss_scaler().check_overflow(parameters_);
        
        // Update loss scale
        mp_manager_.get_loss_scaler().update_scale(has_overflow);
        
        if (has_overflow) {
            std::cout << "Gradient overflow detected, skipping update" << std::endl;
        }
    }
    
    bool should_skip_update() const {
        return mp_manager_.get_loss_scaler().should_skip_update();
    }
    
    MixedPrecisionManager& get_mp_manager() { return mp_manager_; }

private:
    std::vector<Tensor*> parameters_;
    MixedPrecisionManager mp_manager_;
};

} // namespace amp
} // namespace tinytorch