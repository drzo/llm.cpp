#include <iostream>
#include <string>
#include <vector>
#include "../distributed/distributed_training.hpp"
#include "../train_gpt2.cpp"

void print_usage() {
    std::cout << "Distributed Training Tool\n";
    std::cout << "=========================\n\n";
    std::cout << "Usage: mpirun -np <num_processes> distributed_trainer <command> [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  train <model_path> <data_path> <steps> [batch_size] [seq_len]  Start distributed training\n";
    std::cout << "  resume <checkpoint_path> <data_path> <steps>                  Resume from checkpoint\n";
    std::cout << "  test <model_path> <data_path>                                 Test distributed setup\n\n";
    std::cout << "Environment Variables:\n";
    std::cout << "  CUDA_VISIBLE_DEVICES  - GPU devices to use\n";
    std::cout << "  LOCAL_RANK           - Local rank within node\n";
    std::cout << "  WORLD_SIZE           - Total number of processes\n\n";
    std::cout << "Examples:\n";
    std::cout << "  # Single node, 4 GPUs\n";
    std::cout << "  mpirun -np 4 distributed_trainer train model.bin data.bin 1000 64 512\n\n";
    std::cout << "  # Multi-node training\n";
    std::cout << "  mpirun -np 8 -H node1:4,node2:4 distributed_trainer train model.bin data.bin 1000\n\n";
}

void cmd_train(const std::string& model_path, const std::string& data_path, 
              int num_steps, int batch_size = 64, int seq_len = 512) {
    try {
        distributed::DistributedTrainer trainer(model_path, data_path, batch_size, seq_len);
        trainer.train(num_steps);
        trainer.save_model("distributed_model.bin");
        
        std::cout << "Distributed training completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Training failed: " << e.what() << std::endl;
    }
}

void cmd_resume(const std::string& checkpoint_path, const std::string& data_path, int num_steps) {
    try {
        // Load checkpoint and extract model
        GPT2 model;
        CheckpointMetadata metadata;
        
        if (!gpt2_load_checkpoint(&model, checkpoint_path, metadata)) {
            std::cerr << "Could not load checkpoint: " << checkpoint_path << std::endl;
            return;
        }
        
        // Create temporary model file
        std::string temp_model = "temp_model_" + std::to_string(getpid()) + ".bin";
        gpt2_save_checkpoint(&model, temp_model, metadata);
        
        // Start distributed training
        distributed::DistributedTrainer trainer(temp_model, data_path, 
                                               metadata.batch_size, metadata.seq_len);
        trainer.train(num_steps);
        trainer.save_model("resumed_model.bin");
        
        // Clean up
        unlink(temp_model.c_str());
        gpt2_free(&model);
        
        std::cout << "Resumed training completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Resume failed: " << e.what() << std::endl;
    }
}

void cmd_test(const std::string& model_path, const std::string& data_path) {
    try {
        auto config = distributed::DistributedConfig::from_environment();
        
        if (config.is_master()) {
            std::cout << "Testing distributed setup..." << std::endl;
            std::cout << "World size: " << config.world_size << std::endl;
        }
        
        // Test data loading
        distributed::DistributedDataLoader loader(data_path, 64, 512, config);
        loader.next_batch();
        
        if (config.is_master()) {
            std::cout << "Data loading: OK" << std::endl;
        }
        
        // Test model loading
        GPT2 model;
        gpt2_build_from_checkpoint(&model, model_path.c_str());
        
        if (config.is_master()) {
            std::cout << "Model loading: OK" << std::endl;
        }
        
        // Test gradient synchronization
        distributed::DistributedOptimizer optimizer(model.params, config);
        
        if (config.is_master()) {
            std::cout << "Optimizer setup: OK" << std::endl;
        }
        
        // Test forward pass
        gpt2_forward(&model, loader.get_inputs(), loader.get_targets(), 
                    loader.get_local_batch_size(), 512);
        
        if (config.is_master()) {
            std::cout << "Forward pass: OK" << std::endl;
        }
        
        // Test backward pass
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        
        if (config.is_master()) {
            std::cout << "Backward pass: OK" << std::endl;
        }
        
        // Test optimizer step
        optimizer.step(1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, 1);
        
        if (config.is_master()) {
            std::cout << "Optimizer step: OK" << std::endl;
            std::cout << "All tests passed! Ready for distributed training." << std::endl;
        }
        
        gpt2_free(&model);
        
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "train") {
        if (argc < 5) {
            std::cout << "Usage: distributed_trainer train <model_path> <data_path> <steps> [batch_size] [seq_len]" << std::endl;
            return 1;
        }
        
        int num_steps = std::stoi(argv[4]);
        int batch_size = (argc > 5) ? std::stoi(argv[5]) : 64;
        int seq_len = (argc > 6) ? std::stoi(argv[6]) : 512;
        
        cmd_train(argv[2], argv[3], num_steps, batch_size, seq_len);
    }
    else if (command == "resume") {
        if (argc != 5) {
            std::cout << "Usage: distributed_trainer resume <checkpoint_path> <data_path> <steps>" << std::endl;
            return 1;
        }
        
        int num_steps = std::stoi(argv[4]);
        cmd_resume(argv[2], argv[3], num_steps);
    }
    else if (command == "test") {
        if (argc != 4) {
            std::cout << "Usage: distributed_trainer test <model_path> <data_path>" << std::endl;
            return 1;
        }
        
        cmd_test(argv[2], argv[3]);
    }
    else {
        std::cout << "Unknown command: " << command << std::endl;
        print_usage();
        return 1;
    }
    
    return 0;
}