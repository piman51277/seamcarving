# lib
g++ src/lib/cpu/*.cpp -O3 -c -o bin/libcpu.o
ar rcs bin/libcpu.a bin/libcpu.o

# test

g++ src/main.cpp -O3 -Lbin -lcpu  -I/usr/include/SDL2 -D_REENTRANT -lSDL2 -L/usr/lib/x86_64-linux-gnu/ -lpng16 -lSDL2_image -o bin/main
