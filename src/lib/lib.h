// Shared library header file between CPU and GPU versions
#pragma once
#include <cinttypes>
#include <memory>
#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef __ARCH_GPU__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <cuda_fp16.h>
#include <texture_indirect_functions.h>
#endif

namespace SeamCarver
{

  struct ARGB
  {
    uint8_t a;
    uint8_t r;
    uint8_t g;
    uint8_t b;
  };

  union Color
  {
    uint32_t value;
    ARGB argb;
  };

  struct Image
  {
    uint32_t *pixels;
    uint32_t width;
    uint32_t height;

    Image(uint32_t *pixels, uint32_t width, uint32_t height) : pixels(pixels), width(width), height(height) {}
    Image(uint32_t width, uint32_t height) : pixels(new uint32_t[width * height]), width(width), height(height) {}
    Image() : pixels(nullptr), width(0), height(0) {}
    ~Image()
    {
      delete[] pixels;
    }
  };

  class Carver
  {
  private:
// buffers are stored differently on CPU and GPU
#ifdef __ARCH_CPU__
    uint32_t *pixels; // pixels are stored as ARGB 8888
    uint8_t *mask;    // mask for the pixels
    uint32_t *gradient;
    int *seam;     // the next seam to be removed
    uint32_t *buf; // temporary buffer used to avoid reallocation

#endif
#ifdef __ARCH_GPU__
    cudaArray_t pixels; // pixels are stored as ARGB 8888
    cudaArray_t mask;   // mask for the pixels
    cudaArray_t gradient;
    int *seam;       // the next seam to be removed
    cudaArray_t buf; // temporary buffer used to avoid reallocation
#endif

    // these define the actual size of the pixel array
    uint32_t initialWidth;
    uint32_t initialHeight;

    // this is how how many columns remain after carving
    uint32_t currentWidth;

    /**
     * Computes the next seam to be removed.
     */
    void computeSeam();

    /**
     * Removes the last computed seam.
     */
    void removeSeam();

    /**
     * Compute the gradient of the image.
     */
    void computeGradient();

  public:
    Carver(uint32_t *pixels, uint8_t *mask, uint32_t width, uint32_t height);
    Carver(uint32_t *pixels, uint32_t width, uint32_t height);
    ~Carver();

    /**
     * Gets the last used gradient of the image.
     * This copies the gradient into a new buffer and returns it.
     */
    std::unique_ptr<Image> getGradient();

    /**
     * Gets the current state of the image.
     * This copies the pixels into a new buffer and returns it.
     */
    std::unique_ptr<Image> getPixels();

    /**
     * Gets the current width of the image.
     */
    uint32_t width();

    /**
     * Gets the current height of the image.
     */
    uint32_t height();

    /**
     * Removes specified number of seams from the image.
     *
     * @param count the number of seams to remove
     */
    void removeSeams(uint32_t count);
  };
};