# llm.cpp

A high-performance C++ implementation of GPT-2 training featuring a custom tensor library called **tinytorch**. This project is a C++ port of [llm.c](https://github.com/karpathy/llm.c) that maintains simplicity while providing excellent performance through OpenMP parallelization.

## ✨ Features

- **🚀 High Performance**: Optimized C++ implementation with OpenMP support
- **📦 Single-Header Library**: tinytorch is a complete tensor library in one header file
- **🎯 Simple API**: Easy-to-use tensor operations with automatic differentiation
- **🔧 Production Ready**: Comprehensive testing and clean code architecture
- **🌟 Educational**: Clear, readable code perfect for learning deep learning internals

## 🚀 Quick Start

### Prerequisites

- C++ compiler with C++20 support (clang recommended)
- OpenMP (optional but recommended for performance)
  - **macOS**: `brew install libomp`
  - **Ubuntu**: `sudo apt-get install libomp-dev`

### Training GPT-2

```bash
# Download necessary files
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh

# Build and run GPT-2 training
make train_gpt2
OMP_NUM_THREADS=8 ./train_gpt2
```

Expected output:
```
[GPT-2]:
max_seq_len: 1024
vocab_size: 50257
padded_vocab_size: 50304
num_layers: 12
num_heads: 12
channels: 768
Number of Parameters: 124475904
...
step 0 train Loss: 4.67778 (took 7666.71 ms)
step 1 train Loss: 5.19158 (took 7368.44 ms)
...
Checkpoint saved to: checkpoints/checkpoint_step_10.bin
  Step: 10
  Train Loss: 3.24567
  Val Loss: 3.45123
```

### Resume Training from Checkpoint

```bash
# List available checkpoints
./checkpoint_tool list ./checkpoints

# Get checkpoint info
./checkpoint_tool info ./checkpoints/checkpoint_step_100.bin

# Resume training (modify resume_from in main() or use checkpoint_tool)
OMP_NUM_THREADS=8 ./train_gpt2
```

### Running tinytorch Example

```bash
cd example
make
./example
```

## 🧠 tinytorch Library

tinytorch is a lightweight, single-header tensor library that provides:

- **Automatic Differentiation**: Full backward pass support
- **Tensor Operations**: Add, multiply, matrix multiplication, normalization, etc.
- **Neural Network Layers**: Fully connected, attention, layer normalization
- **Activation Functions**: GELU, Softmax, etc.
- **Memory Management**: Efficient tensor context with custom allocator
- **Model Serialization**: Save and load model checkpoints with optimizer state

### Basic Usage

```cpp
#include "tinytorch.hpp"

namespace tt = tinytorch;

int main() {
    // Create tensor context
    tt::TensorContext ctx(1024 * 1024);
    
    // Create tensors
    auto &x = *ctx.NewTensor({32, 64})->RandomNorm();
    auto &W = *ctx.NewTensor({64, 128})->RandomNorm();
    auto &b = *ctx.NewTensor({128})->RandomNorm();
    
    // Forward pass
    auto &y = x.MatMul(W) + b;
    y.Forward();
    
    // Backward pass
    y.Backward();
    
    return 0;
}
```

## 🏗️ Project Structure

```
llm.cpp/
├── tinytorch.hpp           # Single-header tensor library
├── train_gpt2.cpp         # GPT-2 training implementation
├── test_gpt2.cpp          # GPT-2 testing and validation
├── test_tensor.cpp        # tinytorch unit tests
├── checkpoint_manager.hpp  # Checkpoint management utilities
├── tools/
│   └── checkpoint_tool.cpp # Checkpoint inspection and management tool
├── example/
│   ├── tinytorch_example.cpp  # Complete neural network example
│   └── Makefile
├── llmc/                  # Utility headers from llm.c
│   ├── dataloader.h       # Data loading utilities
│   ├── tokenizer.h        # GPT-2 tokenizer
│   ├── rand.h            # Random number generation
│   └── utils.h           # General utilities
├── dev/
│   ├── download_starter_pack.sh  # Download training data
│   └── gen_tensor_test.py        # Test generation utility
└── boost/
    └── ut.hpp            # Unit testing framework
```

## 💾 Model Serialization

The project includes a comprehensive checkpoint system:

### Checkpoint Features

- **Complete State**: Saves model weights, optimizer state, and training metadata
- **Resume Training**: Seamlessly continue training from any checkpoint
- **Automatic Cleanup**: Keeps recent checkpoints and periodic snapshots
- **Metadata Tracking**: Loss curves, training time, configuration details

### Checkpoint Management

```bash
# Build the checkpoint tool
make checkpoint_tool

# List all checkpoints
./checkpoint_tool list ./checkpoints

# Get detailed info about a checkpoint
./checkpoint_tool info ./checkpoints/checkpoint_step_100.bin

# Convert checkpoint to deployment model
./checkpoint_tool convert ./checkpoints/checkpoint_step_100.bin model.bin

# Clean up old checkpoints
./checkpoint_tool cleanup ./checkpoints
```

### Checkpoint Format

Each checkpoint contains:
- **Model weights**: All parameter tensors
- **Optimizer state**: Adam momentum and velocity
- **Training metadata**: Step, loss, learning rate, timestamp
- **Model configuration**: Architecture parameters
- **RNG state**: For reproducible training

## 🧪 Testing

### Run All Tests

```bash
make test_gpt2 test_tensor
./test_gpt2    # Validates against reference implementation
./test_tensor  # Tests tinytorch operations
```

### Performance Testing

```bash
./run_perf.sh  # Runs performance benchmarks
```

## 📊 Neural Network Example

The project includes a complete example of training a neural network to classify data:

```cpp
// Create a simple neural network
auto &x = *ctx.NewTensor({1, 3});        // Input: color, size, weight
auto &y = *ctx.NewTensor({1}, tt::kI32); // Target: class label

// Network parameters
auto &W1 = *ctx.NewTensor({32, 3})->RandomNorm();
auto &b1 = *ctx.NewTensor({32})->RandomNorm();
auto &W2 = *ctx.NewTensor({3, 32})->RandomNorm();
auto &b2 = *ctx.NewTensor({3})->RandomNorm();

// Forward pass
auto &hidden = (x.MatMul(W1) + b1).Gelu();
auto &logits = hidden.MatMul(W2) + b2;
auto &probs = logits.Softmax();
auto &loss = probs.CrossEntropy(y);

// Training loop
for (int step = 0; step < 30000; step++) {
    // Load data
    x.Fill(train_inputs[step % train_size]);
    y.Fill(train_targets[step % train_size]);
    
    // Forward and backward
    loss.Forward();
    loss.ZeroGrad();
    loss.Backward();
    
    // Update parameters (simple SGD)
    for (auto param : {&W1, &b1, &W2, &b2}) {
        auto *weights = (float*)param->data();
        auto *grads = (float*)param->grad()->data();
        for (int i = 0; i < param->NumElements(); i++) {
            weights[i] -= learning_rate * grads[i];
        }
    }
}
```

## 🎯 Key Features

### High-Performance Computing
- **OpenMP Parallelization**: Automatic multi-threading for tensor operations
- **Memory Aligned Operations**: Optimized memory access patterns
- **SIMD Support**: ARM NEON optimizations where available

### Comprehensive Tensor Operations
- **Basic Operations**: Add, multiply, matrix multiplication
- **Advanced Operations**: Transpose, view, split, broadcast
- **Neural Network Layers**: Attention, normalization, activation functions
- **Automatic Differentiation**: Full backward pass computation

### Educational Value
- **Clear Implementation**: Easy-to-understand code structure
- **Comprehensive Testing**: Unit tests for all operations
- **Performance Profiling**: Built-in timing and profiling tools

## 🔧 Build Configuration

The project uses modern C++ features and comprehensive tooling:

- **C++20 Standard**: Modern C++ features and syntax
- **Clang-format**: Consistent code formatting
- **Clang-tidy**: Static analysis and linting
- **Address Sanitizer**: Memory safety checking
- **Optimization Flags**: `-O3 -Ofast` for maximum performance

## 📈 Performance

The implementation achieves excellent performance through:

- **Efficient Memory Layout**: Contiguous tensor storage
- **Parallel Execution**: OpenMP acceleration for compute-heavy operations
- **Cache-Friendly Access**: Optimized memory access patterns
- **Minimal Overhead**: Direct memory operations without excessive abstraction

## 🧮 Mathematics

The library implements key deep learning operations:

- **Matrix Multiplication**: Efficient GEMM implementation
- **Attention Mechanism**: Multi-head self-attention
- **Normalization**: Layer normalization with proper gradient computation
- **Activation Functions**: GELU, Softmax with numerical stability
- **Loss Functions**: Cross-entropy loss with proper gradient flow

## 🤝 Contributing

Contributions are welcome! Please ensure:

1. **Code Quality**: Follow existing style and pass all tests
2. **Documentation**: Update README and comments for new features
3. **Testing**: Add tests for new functionality
4. **Performance**: Maintain or improve performance benchmarks

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) for the original [llm.c](https://github.com/karpathy/llm.c) project
- [GGML](https://github.com/ggerganov/ggml) for inspiration on tensor operations
- The open-source community for tools and libraries used in this project

## 📚 Further Reading

- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [llm.c Repository](https://github.com/karpathy/llm.c)

---

**⭐ Star this repository if you find it helpful!**