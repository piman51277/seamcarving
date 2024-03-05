# Seamcarving

An optimized seam carving library. Supports both CPU and GPU compute under a shared interface. Has a C API for easy integration with other languages. Includes a Python wrapper.

# Building

```
mkdir bin
./compile.sh
```

# Testing

First place a test image called `image.png` in the root directory. Then run:

```
./bin/main_cpu # for CPU compute
./bin/main_gpu # for GPU compute
```

For testing the python wrapper, run:

```
python3 src/py/test.py
```
