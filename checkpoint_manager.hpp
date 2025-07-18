#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <regex>
#include <algorithm>
#include <iostream>

namespace checkpoint_manager {

struct CheckpointInfo {
    std::string path;
    int step;
    double file_size_mb;
    std::string timestamp;
    
    bool operator<(const CheckpointInfo& other) const {
        return step < other.step;
    }
};

class CheckpointManager {
public:
    CheckpointManager(const std::string& base_dir, int keep_last = 3, int keep_every = 100) 
        : base_dir_(base_dir), keep_last_(keep_last), keep_every_(keep_every) {
        std::filesystem::create_directories(base_dir_);
    }
    
    std::string get_checkpoint_path(int step) const {
        return base_dir_ + "/checkpoint_step_" + std::to_string(step) + ".bin";
    }
    
    std::string get_latest_checkpoint() const {
        auto checkpoints = list_checkpoints();
        if (checkpoints.empty()) return "";
        return checkpoints.back().path;
    }
    
    std::vector<CheckpointInfo> list_checkpoints() const {
        std::vector<CheckpointInfo> checkpoints;
        
        if (!std::filesystem::exists(base_dir_)) {
            return checkpoints;
        }
        
        std::regex checkpoint_pattern(R"(checkpoint_step_(\d+)\.bin)");
        std::smatch match;
        
        for (const auto& entry : std::filesystem::directory_iterator(base_dir_)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (std::regex_match(filename, match, checkpoint_pattern)) {
                    int step = std::stoi(match[1].str());
                    double size_mb = entry.file_size() / (1024.0 * 1024.0);
                    
                    auto ftime = std::filesystem::last_write_time(entry);
                    auto time_t = std::chrono::system_clock::to_time_t(
                        std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                            ftime - std::filesystem::file_time_type::clock::now() + 
                            std::chrono::system_clock::now()
                        )
                    );
                    
                    char timestamp[100];
                    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", 
                                 std::localtime(&time_t));
                    
                    checkpoints.push_back({
                        .path = entry.path().string(),
                        .step = step,
                        .file_size_mb = size_mb,
                        .timestamp = timestamp
                    });
                }
            }
        }
        
        std::sort(checkpoints.begin(), checkpoints.end());
        return checkpoints;
    }
    
    void cleanup_old_checkpoints(int current_step) {
        auto checkpoints = list_checkpoints();
        
        for (const auto& checkpoint : checkpoints) {
            bool should_keep = false;
            
            // Keep last N checkpoints
            if (checkpoint.step > current_step - keep_last_) {
                should_keep = true;
            }
            
            // Keep every Nth checkpoint
            if (checkpoint.step % keep_every_ == 0) {
                should_keep = true;
            }
            
            // Keep the very first checkpoint
            if (checkpoint.step == 0) {
                should_keep = true;
            }
            
            if (!should_keep) {
                try {
                    std::filesystem::remove(checkpoint.path);
                    std::cout << "Removed old checkpoint: " << checkpoint.path << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Failed to remove checkpoint " << checkpoint.path 
                              << ": " << e.what() << std::endl;
                }
            }
        }
    }
    
    void print_checkpoint_info() const {
        auto checkpoints = list_checkpoints();
        if (checkpoints.empty()) {
            std::cout << "No checkpoints found in " << base_dir_ << std::endl;
            return;
        }
        
        std::cout << "Available checkpoints:" << std::endl;
        std::cout << "Step\tSize(MB)\tTimestamp" << std::endl;
        std::cout << "----\t--------\t---------" << std::endl;
        
        for (const auto& checkpoint : checkpoints) {
            std::cout << checkpoint.step << "\t" 
                      << std::fixed << std::setprecision(1) << checkpoint.file_size_mb << "\t\t" 
                      << checkpoint.timestamp << std::endl;
        }
    }
    
private:
    std::string base_dir_;
    int keep_last_;
    int keep_every_;
};

} // namespace checkpoint_manager