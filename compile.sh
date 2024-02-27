export CXXFLAGS="-std=c++20 -Ofast -march=native -flto -funroll-loops -flto -fprefetch-loop-arrays"
export CUDAFLAGS="-std=c++20 -O3 -arch=sm_86 -diag-suppress 186"
export CURRENTDIR=$(pwd)

# lib cpu
g++ src/lib/cpu/*.cpp $CXXFLAGS -c -o bin/libcpu.o
ar rcs bin/libcpu.a bin/libcpu.o

#lib gpu
nvcc src/lib/gpu/*.cu $CUDAFLAGS -Xcompiler -fPIC -shared -dc -o bin/gpu.o
ar rcs bin/libgpu.a bin/gpu.o

# test cpu
g++ src/main.cpp $CXXFLAGS -Lbin -lcpu -I/usr/include/SDL2 -D_REENTRANT -lSDL2 -L/usr/lib/x86_64-linux-gnu/ -lpng16 -lSDL2_image -o bin/main_cpu

# test gpu
g++ -c src/main.cpp $CXXFLAGS -I/usr/include/SDL2 -D_REENTRANT -o bin/main_gpu.o -Lbin -lgpu -L/usr/lib/x86_64-linux-gnu/ -lSDL2 -lpng16 -lSDL2_image
nvcc bin/main_gpu.o $CUDAFLAGS -I/usr/include/SDL2 -D_REENTRANT -Lbin -lgpu -L/usr/lib/x86_64-linux-gnu/ -lSDL2 -lpng16 -lSDL2_image -o bin/main_gpu