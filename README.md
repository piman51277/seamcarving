# Seamcarving

An optimized seam carving library. Supports both CPU and GPU compute under a shared interface. Has a C API for easy integration with other languages. Includes a Python wrapper.

## Basic Flask Server

This is a simple implementation of a server using the wrapped library.

```
mkdir bin
./compile.sh
source venv-path/bin/activate
pip install -r requirements.txt
flask --app src/py/app.py run
```

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

# Notes

- To switch the Python wrapper from GPU mode (default) to CPU mode, change the `cdll.LoadLibrary` from `libgpu.so` to `libcpu.so` in `src/py/carver.py`.
