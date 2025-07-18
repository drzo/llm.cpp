#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <cstring>
#include "../tinytorch.hpp"

namespace deployment {

enum class QuantizationType {
    NONE,    // Full precision (FP32)
    INT8,    // 8-bit quantization
    INT16    // 16-bit quantization
};

struct ModelMetadata {
    int magic_number = 20241202;  // Deployment format magic
    int version = 1;
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;
    int num_layers;
    int num_heads;
    int channels;
    QuantizationType quantization;
    float quantization_scale;
    int64_t total_parameters;
    char model_name[64];
    char export_timestamp[64];
};

class ModelExporter {
public:
    explicit ModelExporter(const std::string& model_path) : model_path_(model_path) {}
    
    // Export model for deployment
    bool export_model(const std::string& output_path, 
                     QuantizationType quant_type = QuantizationType::NONE,
                     const std::string& model_name = "GPT2-Custom") {
        try {
            // Load original model
            if (!load_source_model()) {
                return false;
            }
            
            // Create metadata
            ModelMetadata metadata = create_metadata(quant_type, model_name);
            
            // Export with quantization
            std::ofstream file(output_path, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Could not create export file: " << output_path << std::endl;
                return false;
            }
            
            // Write metadata
            file.write(reinterpret_cast<const char*>(&metadata), sizeof(metadata));
            
            // Write quantized parameters
            export_parameters(file, quant_type, metadata.quantization_scale);
            
            file.close();
            
            std::cout << "Model exported successfully to: " << output_path << std::endl;
            std::cout << "  Quantization: " << quantization_type_to_string(quant_type) << std::endl;
            std::cout << "  Model size: " << get_file_size(output_path) / (1024.0 * 1024.0) << " MB" << std::endl;
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Export failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Export optimized model for specific hardware
    bool export_optimized(const std::string& output_path, 
                         const std::string& target_device = "cpu") {
        std::cout << "Exporting optimized model for: " << target_device << std::endl;
        
        QuantizationType quant_type = QuantizationType::NONE;
        if (target_device == "mobile" || target_device == "edge") {
            quant_type = QuantizationType::INT8;
        } else if (target_device == "server") {
            quant_type = QuantizationType::INT16;
        }
        
        return export_model(output_path, quant_type, "GPT2-" + target_device);
    }

private:
    std::string model_path_;
    std::vector<std::vector<float>> parameters_;
    std::vector<std::vector<int>> param_shapes_;
    int max_seq_len_, vocab_size_, padded_vocab_size_;
    int num_layers_, num_heads_, channels_;
    
    bool load_source_model() {
        std::ifstream file(model_path_, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Could not open source model: " << model_path_ << std::endl;
            return false;
        }
        
        // Read model header
        int header[256];
        file.read(reinterpret_cast<char*>(header), sizeof(header));
        
        if (header[0] != 20240326) {
            std::cerr << "Invalid model format" << std::endl;
            return false;
        }
        
        max_seq_len_ = header[2];
        vocab_size_ = header[3];
        num_layers_ = header[4];
        num_heads_ = header[5];
        channels_ = header[6];
        padded_vocab_size_ = header[7];
        
        // Calculate parameter shapes
        calculate_parameter_shapes();
        
        // Load parameters
        parameters_.resize(param_shapes_.size());
        for (size_t i = 0; i < param_shapes_.size(); i++) {
            int num_elements = 1;
            for (int dim : param_shapes_[i]) {
                num_elements *= dim;
            }
            parameters_[i].resize(num_elements);
            file.read(reinterpret_cast<char*>(parameters_[i].data()), 
                     num_elements * sizeof(float));
        }
        
        file.close();
        return true;
    }
    
    void calculate_parameter_shapes() {
        int C = channels_;
        int L = num_layers_;
        int Vp = padded_vocab_size_;
        int maxT = max_seq_len_;
        
        // Embedding parameters
        param_shapes_.push_back({Vp, C});      // wte
        param_shapes_.push_back({maxT, C});    // wpe
        
        // Transformer block parameters (repeated L times)
        std::vector<std::vector<int>> block_shapes = {
            {C},           // ln1w
            {C},           // ln1b
            {3 * C, C},    // qkvw
            {3 * C},       // qkvb
            {C, C},        // attprojw
            {C},           // attprojb
            {C},           // ln2w
            {C},           // ln2b
            {4 * C, C},    // fcw
            {4 * C},       // fcb
            {C, 4 * C},    // fcprojw
            {C}            // fcprojb
        };
        
        for (const auto& shape : block_shapes) {
            for (int l = 0; l < L; l++) {
                param_shapes_.push_back(shape);
            }
        }
        
        // Final layer norm
        param_shapes_.push_back({C});  // lnfw
        param_shapes_.push_back({C});  // lnfb
    }
    
    ModelMetadata create_metadata(QuantizationType quant_type, const std::string& model_name) {
        ModelMetadata metadata;
        metadata.max_seq_len = max_seq_len_;
        metadata.vocab_size = vocab_size_;
        metadata.padded_vocab_size = padded_vocab_size_;
        metadata.num_layers = num_layers_;
        metadata.num_heads = num_heads_;
        metadata.channels = channels_;
        metadata.quantization = quant_type;
        metadata.quantization_scale = calculate_quantization_scale(quant_type);
        
        // Calculate total parameters
        metadata.total_parameters = 0;
        for (const auto& param : parameters_) {
            metadata.total_parameters += param.size();
        }
        
        // Set name and timestamp
        strncpy(metadata.model_name, model_name.c_str(), sizeof(metadata.model_name) - 1);
        auto now = std::time(nullptr);
        auto tm = *std::localtime(&now);
        strftime(metadata.export_timestamp, sizeof(metadata.export_timestamp), 
                "%Y-%m-%d %H:%M:%S", &tm);
        
        return metadata;
    }
    
    float calculate_quantization_scale(QuantizationType quant_type) {
        if (quant_type == QuantizationType::NONE) return 1.0f;
        
        // Find global min/max across all parameters
        float global_min = std::numeric_limits<float>::max();
        float global_max = std::numeric_limits<float>::lowest();
        
        for (const auto& param : parameters_) {
            for (float val : param) {
                global_min = std::min(global_min, val);
                global_max = std::max(global_max, val);
            }
        }
        
        float range = global_max - global_min;
        if (quant_type == QuantizationType::INT8) {
            return range / 255.0f;
        } else if (quant_type == QuantizationType::INT16) {
            return range / 65535.0f;
        }
        
        return 1.0f;
    }
    
    void export_parameters(std::ofstream& file, QuantizationType quant_type, float scale) {
        for (const auto& param : parameters_) {
            if (quant_type == QuantizationType::NONE) {
                // Full precision
                file.write(reinterpret_cast<const char*>(param.data()), 
                          param.size() * sizeof(float));
            } else if (quant_type == QuantizationType::INT8) {
                // 8-bit quantization
                std::vector<uint8_t> quantized(param.size());
                for (size_t i = 0; i < param.size(); i++) {
                    quantized[i] = static_cast<uint8_t>(
                        std::clamp(param[i] / scale, 0.0f, 255.0f));
                }
                file.write(reinterpret_cast<const char*>(quantized.data()), 
                          quantized.size());
            } else if (quant_type == QuantizationType::INT16) {
                // 16-bit quantization
                std::vector<uint16_t> quantized(param.size());
                for (size_t i = 0; i < param.size(); i++) {
                    quantized[i] = static_cast<uint16_t>(
                        std::clamp(param[i] / scale, 0.0f, 65535.0f));
                }
                file.write(reinterpret_cast<const char*>(quantized.data()), 
                          quantized.size() * sizeof(uint16_t));
            }
        }
    }
    
    std::string quantization_type_to_string(QuantizationType type) {
        switch (type) {
            case QuantizationType::NONE: return "FP32";
            case QuantizationType::INT8: return "INT8";
            case QuantizationType::INT16: return "INT16";
            default: return "Unknown";
        }
    }
    
    size_t get_file_size(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        return file.tellg();
    }
};

// Lightweight inference engine
class InferenceEngine {
public:
    explicit InferenceEngine(const std::string& model_path) : model_path_(model_path) {}
    
    bool load_model() {
        std::ifstream file(model_path_, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Could not open model: " << model_path_ << std::endl;
            return false;
        }
        
        // Read metadata
        file.read(reinterpret_cast<char*>(&metadata_), sizeof(metadata_));
        
        if (metadata_.magic_number != 20241202) {
            std::cerr << "Invalid deployment model format" << std::endl;
            return false;
        }
        
        std::cout << "Loading deployment model: " << metadata_.model_name << std::endl;
        std::cout << "  Export time: " << metadata_.export_timestamp << std::endl;
        std::cout << "  Quantization: " << static_cast<int>(metadata_.quantization) << std::endl;
        std::cout << "  Parameters: " << metadata_.total_parameters << std::endl;
        
        // Load parameters based on quantization type
        load_parameters(file);
        
        file.close();
        return true;
    }
    
    // Fast inference interface
    std::vector<float> predict(const std::vector<int>& input_tokens) {
        // Placeholder for inference implementation
        // This would integrate with tinytorch for actual inference
        std::cout << "Running inference on " << input_tokens.size() << " tokens" << std::endl;
        
        // Return mock probabilities for now
        return std::vector<float>(metadata_.vocab_size, 1.0f / metadata_.vocab_size);
    }
    
    const ModelMetadata& get_metadata() const { return metadata_; }

private:
    std::string model_path_;
    ModelMetadata metadata_;
    std::vector<std::vector<float>> parameters_;
    
    void load_parameters(std::ifstream& file) {
        // Implementation depends on quantization type
        // This is a simplified version
        int total_params = metadata_.total_parameters;
        
        if (metadata_.quantization == QuantizationType::NONE) {
            // Load full precision
            std::vector<float> all_params(total_params);
            file.read(reinterpret_cast<char*>(all_params.data()), 
                     total_params * sizeof(float));
            // Split into individual parameter tensors as needed
        } else {
            // Load quantized parameters and dequantize
            std::cout << "Loading quantized parameters..." << std::endl;
            // Implementation for quantized loading
        }
    }
};

} // namespace deployment