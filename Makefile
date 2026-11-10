ARCH ?= native
NVCC ?= nvcc
CXX ?= c++
HOST_SRC = src/scene.cpp src/bvh.cpp src/image.cpp
HEADERS = src/*.h

all: rayforge_gpu

rayforge_gpu: src/gpu_main.cu $(HOST_SRC) $(HEADERS)
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -O3 -o $@ src/gpu_main.cu $(HOST_SRC)

rayforge_cpu: src/cpu_main.cpp $(HOST_SRC) $(HEADERS)
	$(CXX) -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter -pthread -o $@ src/cpu_main.cpp $(HOST_SRC)

clean:
	rm -f rayforge_gpu rayforge_cpu

.PHONY: all clean
