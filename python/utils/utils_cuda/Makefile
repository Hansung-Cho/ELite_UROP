# Compiler
NVCC = nvcc
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -O2 -fdiagnostics-color=always -g -shared -fPIC

# CUDA flags
CUDAFLAGS = -Xcompiler -fPIC --expt-relaxed-constexpr

# Python & pybind11
PYTHON = python3
PYTHON_INC = $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_EXT = $(shell $(PYTHON)-config --extension-suffix)

# CUDA (apt 설치 경로)
CUDA_INCLUDE = /usr/include
CUDA_LIB = /usr/lib/x86_64-linux-gnu

# Include 경로
INCLUDES = -I$(CUDA_INCLUDE) -I/home/rpmsnu/dlpack/include $(PYTHON_INC)
# Linking
LDFLAGS = -L$(CUDA_LIB) -lcudart

# 소스 파일
CU_SRC = kernels.cu
CPP_SRC = bindings.cpp
OBJ = kernels.o
TARGET = mycuda$(PYTHON_EXT)

all: $(TARGET)

# CUDA .cu -> .o
$(OBJ): $(CU_SRC)
	$(NVCC) $(CUDAFLAGS) -c $< -o $@ $(INCLUDES)

# pybind11 확장 모듈 생성
$(TARGET): $(CPP_SRC) $(OBJ)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET) $(OBJ)
