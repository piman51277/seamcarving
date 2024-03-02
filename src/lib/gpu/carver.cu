#include "../lib.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <cuda_fp16.h>

#include <iostream>
#include <iomanip>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

__global__ void rgb2labKernel(SeamCarver::Color *pixels, SeamCarver::CIELAB *lab, uint64_t length)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (uint64_t i = index; i < length; i += stride)
  {
    SeamCarver::ARGB argb = pixels[i].argb;
    float r = (argb.r <= 0.04045) ? (argb.r / 12.92) : pow((argb.r + 0.055) / 1.055, 2.4);
    float g = (argb.g <= 0.04045) ? (argb.g / 12.92) : pow((argb.g + 0.055) / 1.055, 2.4);
    float b = (argb.b <= 0.04045) ? (argb.b / 12.92) : pow((argb.b + 0.055) / 1.055, 2.4);

    float x = r * 0.4124 + g * 0.3576 + b * 0.1805;
    float y = r * 0.2126 + g * 0.7152 + b * 0.0722;
    float z = r * 0.0193 + g * 0.1192 + b * 0.9505;

    x /= 0.95047;
    y /= 1.0;
    z /= 1.08883;

    if (x > 0.008856)
    {
      x = pow(x, 1.0 / 3.0);
    }
    else
    {
      x = 7.787 * x + 16.0 / 116.0;
    }

    if (y > 0.008856)
    {
      y = pow(y, 1.0 / 3.0);
    }
    else
    {
      y = 7.787 * y + 16.0 / 116.0;
    }

    if (z > 0.008856)
    {
      z = pow(z, 1.0 / 3.0);
    }
    else
    {
      z = 7.787 * z + 16.0 / 116.0;
    }

    lab[i].l = 116.0 * y - 16.0;
    lab[i].a = 500.0 * (x - y);
    lab[i].b = 200.0 * (y - z);
  }
}

__device__ float distance(SeamCarver::CIELAB a, SeamCarver::CIELAB b)
{
  float dl = a.l - b.l;
  float da = a.a - b.a;
  float db = a.b - b.b;

  return sqrt(dl * dl + da * da + db * db);
}


//TODO: redo with shared memory
__global__ void gradientKernel(SeamCarver::CIELAB *lab, uint32_t *gradient, uint32_t width, uint32_t height, uint32_t opwidth)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int space = opwidth * height;

  for (uint64_t q = index; q < space; q+=stride)
  {
    int i = q % opwidth;
    int j = q / opwidth;
    
    // 3x3 kernel
    float diffSum = 0;
    float hits = 0;
    for (int k = -1; k <= 1; k++)
    {
      for (int l = -1; l <= 1; l++)
      {
        if (i + k >= 0 && i + k < opwidth && j + l >= 0 && j + l < height)
        {
          hits += 1;
          diffSum += distance(lab[j * width + i], lab[(j + l) * width + i + k]);
        }
      }
    }
    gradient[j * width + i] = (uint32_t)(diffSum / hits * 150.0);
  }
}

__global__ void dpKernel(uint32_t *gradient, uint32_t *buf, uint32_t width, uint32_t height, uint32_t opwidth){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (uint64_t i = index; i < opwidth; i+=stride)
  {
    for(uint64_t j = 1; j < height; j++){
      uint64_t prevRow = (j - 1) * width + i;

      if (i == 0)
      {
        buf[j * width + i] = gradient[j * width + i] + min(buf[prevRow], buf[prevRow + 1]);
      }
      else if (i == opwidth - 1)
      {
        buf[j * width + i] = gradient[j * width + i] + min(buf[prevRow - 1], buf[prevRow]);
      }
      else
      {
        buf[j * width + i] = gradient[j * width + i] + min(min(buf[prevRow - 1], buf[prevRow]), buf[prevRow + 1]);
      }

      //sync threads
      __syncthreads();
    }
  }
}


__global__ void removeKernel(uint32_t *pixels, SeamCarver::CIELAB *lab, uint8_t *mask, int *seam, uint64_t width, uint64_t height, uint64_t currentWidth)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (uint64_t i = index; i < height; i += stride)
  {
    //this is faster than memory copy... why????????
    uint64_t offset = i * width;
    for (uint64_t j = seam[i]; j < currentWidth - 1; j++)
    {
      pixels[offset + j] = pixels[offset + j + 1];
      lab[offset + j] = lab[offset + j + 1];
      mask[offset + j] = mask[offset + j + 1];
    }
  }
}


namespace SeamCarver
{

  Carver::Carver(uint32_t *pixels, u_int8_t *mask, uint32_t width, uint32_t height)
  { 
    cudaMallocManaged(&this->pixels, width * height * sizeof(uint32_t));
    cudaMemcpy(this->pixels, pixels, width * height * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMallocManaged(&this->mask, width * height * sizeof(uint8_t));
    cudaMemcpy(this->mask, mask, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    this->initialWidth = width;
    this->initialHeight = height;
    this->currentWidth = width;
    cudaMallocManaged(&this->lab, width * height * sizeof(CIELAB));
    std::fill(this->lab, this->lab + width * height, CIELAB{0, 0, 0});
    cudaMallocManaged(&this->gradient, width * height * sizeof(uint32_t));
    std::fill(this->gradient, this->gradient + width * height, 0);
    cudaMallocManaged(&this->seam, height * sizeof(int));
    std::fill(this->seam, this->seam + height, 0);
    cudaMallocManaged(&this->buf, width * height * sizeof(uint32_t));
    std::fill(this->buf, this->buf + width * height, 0x00FFFFFF);

    // convert the pixels to CIELAB
    rgb2labKernel<<<16, 256>>>((Color *)this->pixels, this->lab, width * height);
    cudaDeviceSynchronize();
  }

  Carver::Carver(uint32_t *pixels, uint32_t width, uint32_t height)
  {
    cudaMallocManaged(&this->pixels, width * height * sizeof(uint32_t));
    cudaMemcpy(this->pixels, pixels, width * height * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMallocManaged(&this->mask, width * height * sizeof(uint8_t));
    std::fill(this->mask, this->mask + width * height, 0);
    this->initialWidth = width;
    this->initialHeight = height;
    this->currentWidth = width;
    cudaMallocManaged(&this->lab, width * height * sizeof(CIELAB));
    std::fill(this->lab, this->lab + width * height, CIELAB{0, 0, 0});
    cudaMallocManaged(&this->gradient, width * height * sizeof(uint32_t));
    std::fill(this->gradient, this->gradient + width * height, 0);
    cudaMallocManaged(&this->seam, height * sizeof(int));
    std::fill(this->seam, this->seam + height, 0);
    cudaMallocManaged(&this->buf, width * height * sizeof(uint32_t));
    std::fill(this->buf, this->buf + width * height, 0x00FFFFFF);

    // convert the pixels to CIELAB
    rgb2labKernel<<<16, 256>>>((Color *)this->pixels, this->lab, width * height);
    cudaDeviceSynchronize();
  }

  Carver::~Carver()
  {
    cudaFree(this->pixels);
    cudaFree(this->lab);
    cudaFree(this->gradient);
    cudaFree(this->seam);
    cudaFree(this->buf);
  }

  void Carver::computeGradient()
  {
    gradientKernel<<<16, 256>>>(this->lab, this->gradient, this->initialWidth, this->initialHeight, this->currentWidth);
    cudaDeviceSynchronize();
  }

  void Carver::computeSeam()
  {
    // compute the first row of the seam
    for (uint32_t i = 0; i < this->currentWidth; i++)
    {
      this->buf[i] = this->gradient[i];
    }

    dpKernel<<<16, 256>>>(this->gradient, this->buf, this->initialWidth, this->initialHeight, this->currentWidth);
    cudaDeviceSynchronize();

    // find the minimum value in the last row
    uint32_t minIndex = 0; // this is the index relative to row
    for (uint64_t i = 1; i < this->currentWidth; i++)
    {
      if (this->buf[(this->initialHeight - 1) * this->currentWidth + i] < this->buf[(this->initialHeight - 1) * this->currentWidth + minIndex])
      {
        minIndex = i;
      }
    }
    this->seam[this->initialHeight - 1] = minIndex;

    // backtrack to find the seam
    for (int64_t row = this->initialHeight - 2; row >= 0; row--)
    {
      uint64_t searchMin = minIndex == 0 ? 0 : minIndex - 1;
      uint64_t searchMax = minIndex == this->currentWidth - 1 ? this->currentWidth - 1 : minIndex + 1;

      uint32_t minVal = this->buf[row * this->initialWidth + minIndex];
      for (uint64_t i = searchMin; i <= searchMax; i++)
      {
        if (this->buf[row * this->initialWidth + i] < minVal)
        {
          minVal = this->buf[row * this->initialWidth + i];
          minIndex = i;
        }
      }

      this->seam[row] = minIndex;
    }
  }

   void Carver::removeSeam()
  {
    removeKernel<<<8, 256>>>(this->pixels, this->lab, this->mask, this->seam, this->initialWidth, this->initialHeight, this->currentWidth);

    cudaDeviceSynchronize();

    this->currentWidth--;
  }

  std::unique_ptr<Image> Carver::getGradient()
  {
    std::unique_ptr<Image> img(new Image(this->currentWidth, this->initialHeight));
    cudaDeviceSynchronize();
    for (uint64_t i = 0; i < initialHeight; i++)
    {
      uint64_t offset = i * this->initialWidth;
      std::copy(this->gradient + offset, this->gradient + offset + this->currentWidth, img->pixels + i * this->currentWidth);
    }

    return img;
  }

  std::unique_ptr<Image> Carver::getPixels()
  {
    std::unique_ptr<Image> img(new Image(this->currentWidth, this->initialHeight));
    for (uint64_t i = 0; i < initialHeight; i++)
    {
      uint64_t offset = i * this->initialWidth;
      std::copy(this->pixels + offset, this->pixels + offset + this->currentWidth, img->pixels + i * this->currentWidth);
    }
    cudaDeviceSynchronize();
    return img;
  }

  uint32_t Carver::width()
  {
    return this->currentWidth;
  }

  uint32_t Carver::height()
  {
    return this->initialHeight;
  }

  void Carver::removeSeams(uint32_t count)
  {
    if (this->currentWidth - count < 0)
    {
      count = this->currentWidth - 1;
    }

    uint64_t duration1Sum = 0;
    uint64_t duration2Sum = 0;
    uint64_t duration3Sum = 0;

    for (uint32_t i = 0; i < count; i++)
    {
      auto t1 = Clock::now();
      this->computeGradient();
      auto t2 = Clock::now();
      this->computeSeam();
      auto t3 = Clock::now();
      this->removeSeam();
      auto t4 = Clock::now();

      auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
      auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
      auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

      duration1Sum += duration1;
      duration2Sum += duration2;
      duration3Sum += duration3;
    }

    std::cout << std::setprecision(3) << std::fixed; 
    std::cout << "computeGradient: " << duration1Sum / (float)count << "us ";
    std::cout << "computeSeam: " << duration2Sum / (float)count << "us ";
    std::cout << "removeSeam: " << duration3Sum / (float)count << "us" << std::endl;

  }
}
