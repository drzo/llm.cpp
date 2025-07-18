#pragma once

#include <mpi.h>
#include <vector>
#include <memory>
#include <iostream>
#include <cassert>
#include "../tinytorch.hpp"

namespace distributed {

class DistributedConfig {
public:
    int world_size = 1;      // Total number of processes
    int rank = 0;            // Current process rank
    int local_rank = 0;      // Local rank within node
    std::string backend = "nccl";  // Communication backend
    bool use_mixed_precision = false;
    
    bool is_master() const { return rank == 0; }
    
    static DistributedConfig from_environment() {
        DistributedConfig config;
        
        // Initialize MPI
        int provided;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        
        MPI_Comm_size(MPI_COMM_WORLD, &config.world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &config.rank);
        
        // Get local rank from environment or assume single node
        const char* local_rank_env = std::getenv("LOCAL_RANK");
        if (local_rank_env) {
            config.local_rank = std::atoi(local_rank_env);
        } else {
            config.local_rank = config.rank;
        }
        
        if (config.is_master()) {
            std::cout << "Distributed Training Configuration:" << std::endl;
            std::cout << "  World Size: " << config.world_size << std::endl;
            std::cout << "  Backend: " << config.backend << std::endl;
            std::cout << "  Mixed Precision: " << (config.use_mixed_precision ? "Yes" : "No") << std::endl;
        }
        
        return config;
    }
    
    ~DistributedConfig() {
        MPI_Finalize();
    }
};

class AllReduceOp {
public:
    static void all_reduce(std::vector<float>& data, const DistributedConfig& config) {
        if (config.world_size == 1) return;
        
        // Use MPI_Allreduce to sum gradients across all processes
        MPI_Allreduce(MPI_IN_PLACE, data.data(), data.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        
        // Average the gradients
        for (float& val : data) {
            val /= config.world_size;
        }
    }
    
    static void all_reduce_tensor(tinytorch::Tensor* tensor, const DistributedConfig& config) {
        if (config.world_size == 1) return;
        
        auto data = tensor->Flatten();
        all_reduce(data, config);
        
        // Copy back to tensor
        tensor->Fill(data);
    }
    
    static void broadcast(std::vector<float>& data, int root, const DistributedConfig& config) {
        if (config.world_size == 1) return;
        
        MPI_Bcast(data.data(), data.size(), MPI_FLOAT, root, MPI_COMM_WORLD);
    }
};

class DistributedDataLoader {
public:
    DistributedDataLoader(const std::string& data_path, size_t batch_size, size_t seq_len,
                         const DistributedConfig& config)
        : config_(config), global_batch_size_(batch_size), seq_len_(seq_len) {
        
        // Calculate per-process batch size
        local_batch_size_ = batch_size / config.world_size;
        assert(batch_size % config.world_size == 0 && "Batch size must be divisible by world size");
        
        // Initialize data loader for this process
        dataloader_init(&loader_, data_path.c_str(), local_batch_size_, seq_len_, 
                       config.rank, config.world_size, 1);
        
        if (config.is_master()) {
            std::cout << "Distributed Data Loading:" << std::endl;
            std::cout << "  Global Batch Size: " << global_batch_size_ << std::endl;
            std::cout << "  Local Batch Size: " << local_batch_size_ << std::endl;
            std::cout << "  Sequence Length: " << seq_len_ << std::endl;
        }
    }
    
    ~DistributedDataLoader() {
        dataloader_free(&loader_);
    }
    
    void next_batch() {
        dataloader_next_batch(&loader_);
    }
    
    int* get_inputs() { return loader_.inputs; }
    int* get_targets() { return loader_.targets; }
    size_t get_local_batch_size() const { return local_batch_size_; }
    size_t get_global_batch_size() const { return global_batch_size_; }
    
private:
    DataLoader loader_;
    DistributedConfig config_;
    size_t global_batch_size_;
    size_t local_batch_size_;
    size_t seq_len_;
};

class DistributedOptimizer {
public:
    DistributedOptimizer(const std::vector<tinytorch::Tensor*>& parameters,
                        const DistributedConfig& config)
        : parameters_(parameters), config_(config) {
        
        // Initialize optimizer state
        total_parameters_ = 0;
        for (auto* param : parameters_) {
            total_parameters_ += param->NumElements();
        }
        
        m_memory_.resize(total_parameters_, 0.0f);
        v_memory_.resize(total_parameters_, 0.0f);
        
        if (config_.is_master()) {
            std::cout << "Distributed Optimizer initialized with " << total_parameters_ << " parameters" << std::endl;
        }
    }
    
    void step(float learning_rate, float beta1, float beta2, float eps, float weight_decay, int step) {
        // Synchronize gradients across all processes
        synchronize_gradients();
        
        // Apply AdamW update
        apply_adam_update(learning_rate, beta1, beta2, eps, weight_decay, step);
    }
    
    void zero_grad() {
        for (auto* param : parameters_) {
            if (param->grad()) {
                param->grad()->Fill(0.0f);
            }
        }
    }

private:
    std::vector<tinytorch::Tensor*> parameters_;
    DistributedConfig config_;
    std::vector<float> m_memory_;
    std::vector<float> v_memory_;
    size_t total_parameters_;
    
    void synchronize_gradients() {
        if (config_.world_size == 1) return;
        
        // Collect all gradients into a single buffer
        std::vector<float> all_grads;
        all_grads.reserve(total_parameters_);
        
        for (auto* param : parameters_) {
            if (param->grad()) {
                auto grad_data = param->grad()->Flatten();
                all_grads.insert(all_grads.end(), grad_data.begin(), grad_data.end());
            }
        }
        
        // All-reduce gradients
        AllReduceOp::all_reduce(all_grads, config_);
        
        // Copy back to parameter gradients
        size_t offset = 0;
        for (auto* param : parameters_) {
            if (param->grad()) {
                size_t param_size = param->NumElements();
                std::vector<float> grad_slice(all_grads.begin() + offset, 
                                            all_grads.begin() + offset + param_size);
                param->grad()->Fill(grad_slice);
                offset += param_size;
            }
        }
    }
    
    void apply_adam_update(float learning_rate, float beta1, float beta2, 
                          float eps, float weight_decay, int step) {
        size_t idx = 0;
        
        for (auto* param : parameters_) {
            if (!param->grad()) continue;
            
            auto weights = param->Flatten();
            auto grads = param->grad()->Flatten();
            
            for (size_t i = 0; i < weights.size(); i++) {
                float w = weights[i];
                float g = grads[i];
                
                // Update moments
                float m = beta1 * m_memory_[idx] + (1.0f - beta1) * g;
                float v = beta2 * v_memory_[idx] + (1.0f - beta2) * g * g;
                
                // Bias correction
                float m_hat = m / (1.0f - powf(beta1, step));
                float v_hat = v / (1.0f - powf(beta2, step));
                
                // Update parameters
                m_memory_[idx] = m;
                v_memory_[idx] = v;
                weights[i] = w - learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w);
                
                idx++;
            }
            
            // Update parameter tensor
            param->Fill(weights);
        }
    }
};

class DistributedTrainer {
public:
    DistributedTrainer(const std::string& model_path, const std::string& data_path,
                      size_t batch_size, size_t seq_len)
        : config_(DistributedConfig::from_environment()) {
        
        // Load model (all processes load the same initial model)
        gpt2_build_from_checkpoint(&model_, model_path.c_str());
        
        // Create distributed data loader
        train_loader_ = std::make_unique<DistributedDataLoader>(data_path, batch_size, seq_len, config_);
        
        // Create distributed optimizer
        optimizer_ = std::make_unique<DistributedOptimizer>(model_.params, config_);
        
        if (config_.is_master()) {
            std::cout << "Distributed Trainer initialized" << std::endl;
        }
    }
    
    void train(int num_steps, float learning_rate = 1e-4f) {
        if (config_.is_master()) {
            std::cout << "Starting distributed training for " << num_steps << " steps" << std::endl;
        }
        
        for (int step = 0; step < num_steps; step++) {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Load batch
            train_loader_->next_batch();
            
            // Forward pass
            gpt2_forward(&model_, train_loader_->get_inputs(), train_loader_->get_targets(),
                        train_loader_->get_local_batch_size(), train_loader_->get_seq_len());
            
            // Backward pass
            gpt2_zero_grad(&model_);
            gpt2_backward(&model_);
            
            // Distributed optimizer step
            optimizer_->step(learning_rate, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            if (config_.is_master() && step % 10 == 0) {
                std::cout << "Step " << step << " Loss: " << model_.mean_loss 
                         << " Time: " << duration.count() << "ms" << std::endl;
            }
        }
        
        if (config_.is_master()) {
            std::cout << "Distributed training completed" << std::endl;
        }
    }
    
    void save_model(const std::string& output_path) {
        if (config_.is_master()) {
            // Only master process saves the model
            // Implementation would save the current model state
            std::cout << "Saving model to: " << output_path << std::endl;
        }
    }

private:
    DistributedConfig config_;
    GPT2 model_;
    std::unique_ptr<DistributedDataLoader> train_loader_;
    std::unique_ptr<DistributedOptimizer> optimizer_;
};

} // namespace distributed