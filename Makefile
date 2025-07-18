CC = clang
CFLAGS = -O3 -Ofast -std=c++20 -g -fsanitize=address
NVCC = nvcc
NVCCFLAGS = -O3 -std=c++20 -arch=sm_70
LDFLAGS =
LDLIBS = -lm -lc++
INCLUDES =

# CUDA support
ENABLE_CUDA ?= 0
ifeq ($(ENABLE_CUDA), 1)
  CFLAGS += -DENABLE_CUDA
  NVCCFLAGS += -DENABLE_CUDA
  LDLIBS += -lcuda -lcudart -lcublas -lcudnn
  INCLUDES += -I/usr/local/cuda/include
  LDFLAGS += -L/usr/local/cuda/lib64
endif

# Mixed precision support
ENABLE_AMP ?= 0
ifeq ($(ENABLE_AMP), 1)
  CFLAGS += -DENABLE_AMP
  NVCCFLAGS += -DENABLE_AMP
endif

# Check if OpenMP is available
# This is done by attempting to compile an empty file with OpenMP flags
# OpenMP makes the code a lot faster so I advise installing it
# e.g. on MacOS: brew install libomp
# e.g. on Ubuntu: sudo apt-get install libomp-dev
# later, run the program by prepending the number of threads, e.g.: OMP_NUM_THREADS=8 ./gpt2
ifeq ($(shell echo | $(CC) -Xpreprocessor -fopenmp -x c -E - > /dev/null 2>&1; echo $$?), 0)
  ifeq ($(shell uname), Darwin)
    # macOS with Homebrew
    CFLAGS += -Xclang -fopenmp
    LDFLAGS += -L/opt/homebrew/opt/libomp/lib
    LDLIBS += -lomp
    INCLUDES += -I/opt/homebrew/opt/libomp/include
  else
    # Ubuntu or other Linux distributions
    CFLAGS += -fopenmp
    LDLIBS += -lgomp
  endif
  $(info NICE Compiling with OpenMP support)
else
  $(warning OOPS Compiling without OpenMP support)
endif

# PHONY means these targets will always be executed
.PHONY: all train_gpt2 test_gpt2 test_tensor checkpoint_tool model_export_tool distributed_trainer cuda_kernels

# default target is all
all: train_gpt2 test_gpt2 test_tensor checkpoint_tool model_export_tool distributed_trainer

# CUDA kernels compilation
cuda_kernels:
ifeq ($(ENABLE_CUDA), 1)
	$(NVCC) $(NVCCFLAGS) -c cuda/cuda_kernels.cu -o cuda_kernels.o
endif

train_gpt2: train_gpt2.cpp
ifeq ($(ENABLE_CUDA), 1)
	$(MAKE) cuda_kernels
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< cuda_kernels.o $(LDLIBS) -o $@
else
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@
endif

test_gpt2: test_gpt2.cpp
ifeq ($(ENABLE_CUDA), 1)
	$(MAKE) cuda_kernels
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< cuda_kernels.o $(LDLIBS) -o $@
else
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@
endif

test_tensor: test_tensor.cpp
ifeq ($(ENABLE_CUDA), 1)
	$(MAKE) cuda_kernels
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ cuda_kernels.o ${LDLIBS} -o $@
else
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ ${LDLIBS} -o $@
endif
	
checkpoint_tool: tools/checkpoint_tool.cpp
ifeq ($(ENABLE_CUDA), 1)
	$(MAKE) cuda_kernels
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< cuda_kernels.o $(LDLIBS) -o $@
else
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@
endif
	
model_export_tool: tools/model_export_tool.cpp
ifeq ($(ENABLE_CUDA), 1)
	$(MAKE) cuda_kernels
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< cuda_kernels.o $(LDLIBS) -o $@
else
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -o $@
endif
	
distributed_trainer: tools/distributed_trainer.cpp
ifeq ($(ENABLE_CUDA), 1)
	$(MAKE) cuda_kernels
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< cuda_kernels.o $(LDLIBS) -lmpi -o $@
else
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $< $(LDLIBS) -lmpi -o $@
endif
	
clean:
	rm -f train_gpt2 test_gpt2 test_tensor checkpoint_tool model_export_tool distributed_trainer cuda_kernels.o

# Convenience targets
cuda: ENABLE_CUDA=1
cuda: all

amp: ENABLE_AMP=1
amp: all

cuda-amp: ENABLE_CUDA=1
cuda-amp: ENABLE_AMP=1
cuda-amp: all
