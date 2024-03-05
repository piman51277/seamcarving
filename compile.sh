
export CXXFLAGS="-std=c++20 -Ofast -march=native -flto -funroll-loops -flto -fprefetch-loop-arrays"
export CUDAFLAGS="-std=c++20 -O3 -diag-suppress 186"

# lib cpu
g++ lib/cpu/carver.cpp lib/wrap.cpp $CXXFLAGS -fPIC -shared -o bin/libcpu.so

#lib gpu
nvcc lib/gpu/carver.cu lib/wrap.cpp $CUDAFLAGS -Xcompiler -fPIC -shared -o bin/libgpu.so

export LD_LIBRARY_PATH=$(pwd)/bin
export LINKOPT="-I/usr/include/SDL2 -D_REENTRANT -Lbin -L/usr/lib/x86_64-linux-gnu/ -lSDL2 -lpng16 -lSDL2_image"

# test scripts
g++ src/cpp/main.cpp $CXXFLAGS -Wl,-rpath=$LD_LIBRARY_PATH $LINKOPT -lcpu -o bin/main_cpu
g++ -c src/cpp/main.cpp $CXXFLAGS -Wl,-rpath=$LD_LIBRARY_PATH $LINKOPT -o bin/main_gpu.o 
nvcc bin/main_gpu.o $CUDAFLAGS -Xlinker -rpath=$LD_LIBRARY_PATH $LINKOPT -lgpu -o bin/main_gpu
rm bin/main_gpu.o