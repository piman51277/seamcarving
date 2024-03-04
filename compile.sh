
export CXXFLAGS="-std=c++20 -Ofast -march=native -flto -funroll-loops -flto -fprefetch-loop-arrays"
export CUDAFLAGS="-std=c++20  -O3 -diag-suppress 186"

# lib cpu
g++ lib/cpu/carver.cpp lib/wrap.cpp $CXXFLAGS -fPIC -shared -o bin/libcpu.so

#lib gpu
nvcc lib/gpu/carver.cu lib/wrap.cpp $CUDAFLAGS -Xcompiler -fPIC -shared -o bin/libgpu.so