#include <iostream>
#include <string>
#include <vector>
#include "../deployment/model_export.hpp"
#include "../train_gpt2.cpp"

void print_usage() {
    std::cout << "Model Export Tool\n";
    std::cout << "================\n\n";
    std::cout << "Usage: model_export_tool <command> [options]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  export <model_path> <output_path> [quantization]  Export model for deployment\n";
    std::cout << "  optimize <model_path> <output_path> <device>      Export optimized for device\n";
    std::cout << "  info <deployment_model_path>                     Show deployment model info\n";
    std::cout << "  benchmark <deployment_model_path>                Benchmark inference speed\n\n";
    std::cout << "Quantization options:\n";
    std::cout << "  fp32    - Full precision (default)\n";
    std::cout << "  int16   - 16-bit quantization\n";
    std::cout << "  int8    - 8-bit quantization\n\n";
    std::cout << "Device options:\n";
    std::cout << "  cpu     - CPU optimized\n";
    std::cout << "  mobile  - Mobile/edge optimized (INT8)\n";
    std::cout << "  server  - Server optimized (INT16)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  model_export_tool export model.bin deploy.bin int8\n";
    std::cout << "  model_export_tool optimize model.bin mobile.bin mobile\n";
    std::cout << "  model_export_tool info deploy.bin\n";
}

deployment::QuantizationType parse_quantization(const std::string& quant_str) {
    if (quant_str == "fp32") return deployment::QuantizationType::NONE;
    if (quant_str == "int16") return deployment::QuantizationType::INT16;
    if (quant_str == "int8") return deployment::QuantizationType::INT8;
    return deployment::QuantizationType::NONE;
}

void cmd_export(const std::string& model_path, const std::string& output_path, 
               const std::string& quant_str = "fp32") {
    deployment::ModelExporter exporter(model_path);
    deployment::QuantizationType quant_type = parse_quantization(quant_str);
    
    if (exporter.export_model(output_path, quant_type)) {
        std::cout << "Export completed successfully!" << std::endl;
    } else {
        std::cerr << "Export failed!" << std::endl;
    }
}

void cmd_optimize(const std::string& model_path, const std::string& output_path, 
                 const std::string& device) {
    deployment::ModelExporter exporter(model_path);
    
    if (exporter.export_optimized(output_path, device)) {
        std::cout << "Optimization completed successfully!" << std::endl;
    } else {
        std::cerr << "Optimization failed!" << std::endl;
    }
}

void cmd_info(const std::string& deployment_model_path) {
    deployment::InferenceEngine engine(deployment_model_path);
    
    if (engine.load_model()) {
        const auto& metadata = engine.get_metadata();
        
        std::cout << "Deployment Model Information" << std::endl;
        std::cout << "============================" << std::endl;
        std::cout << "Model Name: " << metadata.model_name << std::endl;
        std::cout << "Export Time: " << metadata.export_timestamp << std::endl;
        std::cout << "Version: " << metadata.version << std::endl;
        std::cout << "Architecture:" << std::endl;
        std::cout << "  Max Seq Len: " << metadata.max_seq_len << std::endl;
        std::cout << "  Vocab Size: " << metadata.vocab_size << std::endl;
        std::cout << "  Padded Vocab Size: " << metadata.padded_vocab_size << std::endl;
        std::cout << "  Num Layers: " << metadata.num_layers << std::endl;
        std::cout << "  Num Heads: " << metadata.num_heads << std::endl;
        std::cout << "  Channels: " << metadata.channels << std::endl;
        std::cout << "Quantization:" << std::endl;
        std::cout << "  Type: " << static_cast<int>(metadata.quantization) << std::endl;
        std::cout << "  Scale: " << metadata.quantization_scale << std::endl;
        std::cout << "Parameters: " << metadata.total_parameters << std::endl;
    } else {
        std::cerr << "Could not load deployment model!" << std::endl;
    }
}

void cmd_benchmark(const std::string& deployment_model_path) {
    deployment::InferenceEngine engine(deployment_model_path);
    
    if (!engine.load_model()) {
        std::cerr << "Could not load deployment model!" << std::endl;
        return;
    }
    
    const auto& metadata = engine.get_metadata();
    std::cout << "Benchmarking model: " << metadata.model_name << std::endl;
    
    // Create test input
    std::vector<int> test_tokens(64);  // 64 tokens
    for (int i = 0; i < 64; i++) {
        test_tokens[i] = i % metadata.vocab_size;
    }
    
    // Warm up
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 5; i++) {
        engine.predict(test_tokens);
    }
    
    // Benchmark
    std::cout << "Running benchmark..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    int num_runs = 100;
    for (int i = 0; i < num_runs; i++) {
        auto probs = engine.predict(test_tokens);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    float avg_time = duration.count() / static_cast<float>(num_runs);
    float tokens_per_second = (64 * 1000.0f) / avg_time;
    
    std::cout << "Benchmark Results:" << std::endl;
    std::cout << "  Average time per batch: " << avg_time << " ms" << std::endl;
    std::cout << "  Tokens per second: " << tokens_per_second << std::endl;
    std::cout << "  Runs: " << num_runs << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "export") {
        if (argc < 4) {
            std::cout << "Usage: model_export_tool export <model_path> <output_path> [quantization]" << std::endl;
            return 1;
        }
        std::string quant_str = (argc > 4) ? argv[4] : "fp32";
        cmd_export(argv[2], argv[3], quant_str);
    }
    else if (command == "optimize") {
        if (argc != 5) {
            std::cout << "Usage: model_export_tool optimize <model_path> <output_path> <device>" << std::endl;
            return 1;
        }
        cmd_optimize(argv[2], argv[3], argv[4]);
    }
    else if (command == "info") {
        if (argc != 3) {
            std::cout << "Usage: model_export_tool info <deployment_model_path>" << std::endl;
            return 1;
        }
        cmd_info(argv[2]);
    }
    else if (command == "benchmark") {
        if (argc != 3) {
            std::cout << "Usage: model_export_tool benchmark <deployment_model_path>" << std::endl;
            return 1;
        }
        cmd_benchmark(argv[2]);
    }
    else {
        std::cout << "Unknown command: " << command << std::endl;
        print_usage();
        return 1;
    }
    
    return 0;
}