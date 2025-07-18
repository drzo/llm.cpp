#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include "../train_gpt2.cpp"
#include "../checkpoint_manager.hpp"

void print_usage() {
    std::cout << "GPT-2 Checkpoint Tool\n";
    std::cout << "====================\n\n";
    std::cout << "Usage: checkpoint_tool <command> [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  list <checkpoint_dir>              List all checkpoints\n";
    std::cout << "  info <checkpoint_path>             Show checkpoint details\n";
    std::cout << "  convert <checkpoint_path> <output> Convert checkpoint to model\n";
    std::cout << "  cleanup <checkpoint_dir>           Clean up old checkpoints\n";
    std::cout << "  resume <checkpoint_path>           Show resume command\n\n";
    std::cout << "Examples:\n";
    std::cout << "  checkpoint_tool list ./checkpoints\n";
    std::cout << "  checkpoint_tool info ./checkpoints/checkpoint_step_100.bin\n";
    std::cout << "  checkpoint_tool convert ./checkpoints/checkpoint_step_100.bin model.bin\n";
}

void cmd_list(const std::string& checkpoint_dir) {
    checkpoint_manager::CheckpointManager manager(checkpoint_dir);
    manager.print_checkpoint_info();
}

void cmd_info(const std::string& checkpoint_path) {
    GPT2 model;
    CheckpointMetadata metadata;
    
    try {
        if (gpt2_load_checkpoint(&model, checkpoint_path, metadata)) {
            std::cout << "Checkpoint Information" << std::endl;
            std::cout << "=====================" << std::endl;
            std::cout << "Path: " << checkpoint_path << std::endl;
            std::cout << "Training Step: " << metadata.training_step << std::endl;
            std::cout << "Learning Rate: " << metadata.learning_rate << std::endl;
            std::cout << "Train Loss: " << metadata.train_loss << std::endl;
            std::cout << "Val Loss: " << metadata.val_loss << std::endl;
            std::cout << "Total Time: " << metadata.total_time_ms / 1000.0 << " seconds" << std::endl;
            std::cout << "Batch Size: " << metadata.batch_size << std::endl;
            std::cout << "Sequence Length: " << metadata.seq_len << std::endl;
            std::cout << "Dataset: " << metadata.dataset_path << std::endl;
            std::cout << "Timestamp: " << metadata.timestamp << std::endl;
            std::cout << "\nModel Configuration:" << std::endl;
            std::cout << "  Max Seq Len: " << model.config.max_seq_len << std::endl;
            std::cout << "  Vocab Size: " << model.config.vocab_size << std::endl;
            std::cout << "  Padded Vocab Size: " << model.config.padded_vocab_size << std::endl;
            std::cout << "  Num Layers: " << model.config.num_layers << std::endl;
            std::cout << "  Num Heads: " << model.config.num_heads << std::endl;
            std::cout << "  Channels: " << model.config.channels << std::endl;
            std::cout << "  Parameters: " << model.num_parameters << std::endl;
            
            gpt2_free(&model);
        } else {
            std::cout << "Could not load checkpoint: " << checkpoint_path << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
}

void cmd_convert(const std::string& checkpoint_path, const std::string& output_path) {
    GPT2 model;
    CheckpointMetadata metadata;
    
    try {
        if (gpt2_load_checkpoint(&model, checkpoint_path, metadata)) {
            // Save as standard model format (without optimizer state)
            FILE* output_file = fopen(output_path.c_str(), "wb");
            if (output_file == nullptr) {
                throw std::runtime_error("Could not create output file: " + output_path);
            }
            
            // Write model header
            int model_header[256] = {0};
            model_header[0] = 20240326;  // model magic number
            model_header[1] = 3;         // model version
            model_header[2] = model.config.max_seq_len;
            model_header[3] = model.config.vocab_size;
            model_header[4] = model.config.num_layers;
            model_header[5] = model.config.num_heads;
            model_header[6] = model.config.channels;
            model_header[7] = model.config.padded_vocab_size;
            fwrite(model_header, sizeof(int), 256, output_file);
            
            // Write model parameters
            for (auto param : model.params) {
                fwrite(param->data(), sizeof(float), param->NumElements(), output_file);
            }
            
            fclose(output_file);
            std::cout << "Model converted and saved to: " << output_path << std::endl;
            
            gpt2_free(&model);
        } else {
            std::cout << "Could not load checkpoint: " << checkpoint_path << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
}

void cmd_cleanup(const std::string& checkpoint_dir) {
    checkpoint_manager::CheckpointManager manager(checkpoint_dir);
    auto checkpoints = manager.list_checkpoints();
    
    if (checkpoints.empty()) {
        std::cout << "No checkpoints found in " << checkpoint_dir << std::endl;
        return;
    }
    
    int latest_step = checkpoints.back().step;
    std::cout << "Cleaning up old checkpoints (keeping last 3 and every 100th)..." << std::endl;
    manager.cleanup_old_checkpoints(latest_step);
    std::cout << "Cleanup completed." << std::endl;
}

void cmd_resume(const std::string& checkpoint_path) {
    std::cout << "To resume training from this checkpoint, use:" << std::endl;
    std::cout << "  ./train_gpt2 --resume " << checkpoint_path << std::endl;
    std::cout << "\nOr modify the resume_from variable in main():" << std::endl;
    std::cout << "  const std::string resume_from = \"" << checkpoint_path << "\";" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "list") {
        if (argc != 3) {
            std::cout << "Usage: checkpoint_tool list <checkpoint_dir>" << std::endl;
            return 1;
        }
        cmd_list(argv[2]);
    }
    else if (command == "info") {
        if (argc != 3) {
            std::cout << "Usage: checkpoint_tool info <checkpoint_path>" << std::endl;
            return 1;
        }
        cmd_info(argv[2]);
    }
    else if (command == "convert") {
        if (argc != 4) {
            std::cout << "Usage: checkpoint_tool convert <checkpoint_path> <output_path>" << std::endl;
            return 1;
        }
        cmd_convert(argv[2], argv[3]);
    }
    else if (command == "cleanup") {
        if (argc != 3) {
            std::cout << "Usage: checkpoint_tool cleanup <checkpoint_dir>" << std::endl;
            return 1;
        }
        cmd_cleanup(argv[2]);
    }
    else if (command == "resume") {
        if (argc != 3) {
            std::cout << "Usage: checkpoint_tool resume <checkpoint_path>" << std::endl;
            return 1;
        }
        cmd_resume(argv[2]);
    }
    else {
        std::cout << "Unknown command: " << command << std::endl;
        print_usage();
        return 1;
    }
    
    return 0;
}