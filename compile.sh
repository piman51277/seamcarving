export CXXFLAGS="-std=c++20 -Ofast -march=native -flto -funroll-loops -flto -fprefetch-loop-arrays"

# lib
g++ src/lib/cpu/*.cpp $CXXFLAGS -c -o bin/libcpu.o
ar rcs bin/libcpu.a bin/libcpu.o

# test
g++ src/main.cpp $CXXFLAGS -Lbin -lcpu -I/usr/include/SDL2 -D_REENTRANT -lSDL2 -L/usr/lib/x86_64-linux-gnu/ -lpng16 -lSDL2_image -o bin/main_cpu
