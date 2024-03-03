#define __ARCH_GPU__
#include "../lib.h"

#include <iostream>
#include <iomanip>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

cudaTextureObject_t createTexture(cudaArray *cuArray)
{
  // Create a cudaResourceDesc
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  // Create a cudaTextureDesc
  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  texDesc.filterMode = cudaFilterModePoint;
  // Create the texture object
  cudaTextureObject_t texObj;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

  return texObj;
}

cudaSurfaceObject_t createSurface(cudaArray *cuArray)
{
  // Create a cudaResourceDesc
  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  // Create the surface object
  cudaSurfaceObject_t surfObj;
  cudaCreateSurfaceObject(&surfObj, &resDesc);

  return surfObj;
}

__device__ uint32_t distance(SeamCarver::Color a, SeamCarver::Color b)
{
  uint8_t rd = abs(a.argb.r - b.argb.r);
  uint8_t gd = abs(a.argb.g - b.argb.g);
  uint8_t bd = abs(a.argb.b - b.argb.b);

  return __fsqrt_rn(rd * rd + gd * gd + bd * bd);
}

// TODO: redo with shared memory
__global__ void gradientKernel(cudaTextureObject_t pixels, cudaSurfaceObject_t gradient, cudaTextureObject_t mask, uint32_t width, uint32_t height, uint32_t opwidth)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int space = opwidth * height;

  for (uint64_t q = index; q < space; q += stride)
  {
    int i = q % opwidth;
    int j = q / opwidth;

    if (tex2D<uint8_t>(mask, i, j) == 1)
    {
      surf2Dwrite(0xFFFF0000, gradient, i * sizeof(uint32_t), j);
      continue;
    }

    // 3x3 kernel
    uint32_t diffSum = 0;
    float hits = 0;
    for (int k = -1; k <= 1; k++)
    {
      for (int l = -1; l <= 1; l++)
      {
        if (i + k >= 0 && i + k < opwidth && j + l >= 0 && j + l < height)
        {
          hits += 1;
          diffSum += distance((SeamCarver::Color)tex2D<uint32_t>(pixels, i, j), (SeamCarver::Color)tex2D<uint32_t>(pixels, i + k, j + l));
        }
      }
    }
    surf2Dwrite((uint32_t)(diffSum / hits), gradient, i * sizeof(uint32_t), j);
  }
}


//FIXME: incorrect results around thread block boundaries
__global__ void dpKernel(cudaTextureObject_t gradient, cudaSurfaceObject_t buf, uint32_t width, uint32_t height, uint32_t opwidth)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // compute the first row of the seam
  for (uint64_t i = index; i < opwidth; i += stride)
  {
    surf2Dwrite(tex2D<uint32_t>(gradient, i, 0), buf, i * sizeof(uint32_t), 0);
  }

  __syncthreads();

  for (uint64_t i = index; i < opwidth; i += stride)
  {
    for (uint64_t j = 1; j < height; j++)
    {

      if (i == 0)
      {
        surf2Dwrite(tex2D<uint32_t>(gradient, i, j) + min(tex2D<uint32_t>(buf, i, j - 1), tex2D<uint32_t>(buf, i + 1, j - 1)), buf, i * sizeof(uint32_t), j);
      }
      else if (i == opwidth - 1)
      {
        surf2Dwrite(tex2D<uint32_t>(gradient, i, j) + min(tex2D<uint32_t>(buf, i - 1, j - 1), tex2D<uint32_t>(buf, i, j - 1)), buf, i * sizeof(uint32_t), j);
      }
      else
      {
        surf2Dwrite(tex2D<uint32_t>(gradient, i, j) + min(min(tex2D<uint32_t>(buf, i - 1, j - 1), tex2D<uint32_t>(buf, i, j - 1)), tex2D<uint32_t>(buf, i + 1, j - 1)), buf, i * sizeof(uint32_t), j);
      }

      __syncthreads();
    }
  }
}

__global__ void findSeamKernel(cudaSurfaceObject_t buf, int *seam, uint32_t width, uint32_t height, uint32_t opwidth)
{
  //  find the minimum value in the last row
  uint32_t minIndex = 0; // this is the index relative to row
  for (uint64_t i = 1; i < opwidth; i++)
  {
    if (tex2D<uint32_t>(buf, i, height - 1) < tex2D<uint32_t>(buf, minIndex, height - 1))
    {
      minIndex = i;
    }
  }
  seam[height - 1] = minIndex;

  // backtrack to find the seam
  for (int64_t row = height - 2; row >= 0; row--)
  {
    uint64_t searchMin = minIndex == 0 ? 0 : minIndex - 1;
    uint64_t searchMax = minIndex == opwidth - 1 ? opwidth - 1 : minIndex + 1;

    uint32_t minVal = tex2D<uint32_t>(buf, minIndex, row);
    for (uint64_t i = searchMin; i <= searchMax; i++)
    {
      if (tex2D<uint32_t>(buf, i, row) < minVal)
      {
        minVal = tex2D<uint32_t>(buf, i, row);
        minIndex = i;
      }
    }

    seam[row] = minIndex;
  }
}

__global__ void removeKernel(cudaSurfaceObject_t pixels, cudaTextureObject_t mask, int *seam, uint64_t width, uint64_t height, uint64_t currentWidth)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (uint64_t i = index; i < height; i += stride)
  {
    for (uint64_t j = seam[i]; j < currentWidth - 1; j++)
    {
      surf2Dwrite(tex2D<uint32_t>(pixels, j + 1, i), pixels, j * sizeof(uint32_t), i);
      surf2Dwrite(tex2D<uint8_t>(mask, j + 1, i), mask, j, i);
    }
  }
}

__global__ void copySurfaceArrKernel(cudaSurfaceObject_t src, uint32_t *dst, uint64_t width, uint64_t height, uint64_t opwidth)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (uint64_t i = index; i < height; i += stride)
  {
    for (uint64_t j = 0; j < opwidth; j++)
    {
      dst[i * width + j] = surf2Dread<uint32_t>(src, j * sizeof(uint32_t), i);
    }
  }
}

namespace SeamCarver
{

  Carver::Carver(uint32_t *pixels, u_int8_t *mask, uint32_t width, uint32_t height)
  {
    cudaChannelFormatDesc channelDescUint32 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaMallocArray(&this->pixels, &channelDescUint32, width, height);
    cudaMemcpy2DToArray(this->pixels, 0, 0, pixels, width * sizeof(uint32_t), width * sizeof(uint32_t), height, cudaMemcpyHostToDevice);

    cudaChannelFormatDesc channelDescMask = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaMallocArray(&this->mask, &channelDescMask, width, height);
    cudaMemcpy2DToArray(this->mask, 0, 0, mask, width * sizeof(uint8_t), width * sizeof(uint8_t), height, cudaMemcpyHostToDevice);

    this->initialWidth = width;
    this->initialHeight = height;
    this->currentWidth = width;

    // we don't need to fill these with 0s because we will overwrite
    cudaMallocArray(&this->gradient, &channelDescUint32, width, height);
    cudaMallocManaged(&this->seam, height * sizeof(int));
    cudaMallocArray(&this->buf, &channelDescUint32, width, height);
  }

  Carver::Carver(uint32_t *pixels, uint32_t width, uint32_t height)
  {
    cudaChannelFormatDesc channelDescUint32 = cudaCreateChannelDesc<uint32_t>();
    cudaMallocArray(&this->pixels, &channelDescUint32, width, height, cudaArraySurfaceLoadStore);
    cudaMemcpy2DToArray(this->pixels, 0, 0, pixels, width * sizeof(uint32_t), width * sizeof(uint32_t), height, cudaMemcpyHostToDevice);

    this->initialWidth = width;
    this->initialHeight = height;
    this->currentWidth = width;

    cudaChannelFormatDesc channelDescMask = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaMallocArray(&this->mask, &channelDescMask, width, height, cudaArraySurfaceLoadStore);

    // idk why we have to go through this song and dance but oh well
    uint32_t *mask;
    cudaMalloc(&mask, width * height * sizeof(uint8_t));
    cudaMemset(mask, 0, width * height * sizeof(uint8_t));
    cudaMemcpy2DToArray(this->mask, 0, 0, mask, width * sizeof(uint8_t), width * sizeof(uint8_t), height, cudaMemcpyHostToDevice);
    cudaFree(mask);

    // we don't need to fill these with 0s because we will overwrite
    cudaMallocArray(&this->gradient, &channelDescUint32, width, height, cudaArraySurfaceLoadStore);
    cudaMallocManaged(&this->seam, height * sizeof(int));
    cudaMallocArray(&this->buf, &channelDescUint32, width, height, cudaArraySurfaceLoadStore);
  }

  Carver::~Carver()
  {
    cudaFree(this->pixels);
    cudaFree(this->gradient);
    cudaFree(this->seam);
    cudaFree(this->buf);
  }

  void Carver::computeGradient()
  {
    // pixels will be read-only for gradient kernel
    auto pix = createTexture(this->pixels);

    // mask will be read-only for gradient kernel
    auto mask = createTexture(this->mask);

    // gradient will be written to
    auto grad = createSurface(this->gradient);

    gradientKernel<<<256, 1024>>>(pix, grad, mask, this->initialWidth, this->initialHeight, this->currentWidth);
    cudaDeviceSynchronize();

    cudaDestroyTextureObject(pix);
    cudaDestroyTextureObject(mask);
    cudaDestroySurfaceObject(grad);
  }

  void Carver::computeSeam()
  {
    // gradient will be read-only for dp kernel
    auto grad = createTexture(this->gradient);

    // buf will be written to
    auto buf = createSurface(this->buf);

    // fewer thread blocks, less distortion, since this is an approximation
    dpKernel<<<256, 256>>>(grad, buf, this->initialWidth, this->initialHeight, this->currentWidth);
    cudaDeviceSynchronize();

    findSeamKernel<<<1, 1>>>(buf, this->seam, this->initialWidth, this->initialHeight, this->currentWidth);
    cudaDeviceSynchronize();

    cudaDestroyTextureObject(grad);
    cudaDestroySurfaceObject(buf);
  }

  void Carver::removeSeam()
  {
    auto pix = createSurface(this->pixels);
    auto mask = createTexture(this->mask);

    removeKernel<<<256, 32>>>(pix, mask, this->seam, this->initialWidth, this->initialHeight, this->currentWidth);
    cudaDeviceSynchronize();

    this->currentWidth--;
    cudaDestroySurfaceObject(pix);
    cudaDestroyTextureObject(mask);
  }

  std::unique_ptr<Image> Carver::getGradient()
  {
    auto pix = createSurface(this->buf);
    uint32_t *pixels;
    cudaMallocManaged(&pixels, this->currentWidth * this->initialHeight * sizeof(uint32_t));
    copySurfaceArrKernel<<<64, 64>>>(pix, pixels, this->currentWidth, this->initialHeight, this->currentWidth);
    cudaDestroyTextureObject(pix);
    cudaDeviceSynchronize();

    std::unique_ptr<Image> img(new Image(this->currentWidth, this->initialHeight));
    std::copy(pixels, pixels + this->currentWidth * this->initialHeight, img->pixels);
    cudaFree(pixels);
    return img;
  }

  std::unique_ptr<Image> Carver::getPixels()
  {
    auto pix = createSurface(this->pixels);
    uint32_t *pixels;
    cudaMallocManaged(&pixels, this->currentWidth * this->initialHeight * sizeof(uint32_t));
    copySurfaceArrKernel<<<64, 64>>>(pix, pixels, this->currentWidth, this->initialHeight, this->currentWidth);
    cudaDestroyTextureObject(pix);
    cudaDeviceSynchronize();

    std::unique_ptr<Image> img(new Image(this->currentWidth, this->initialHeight));
    std::copy(pixels, pixels + this->currentWidth * this->initialHeight, img->pixels);
    cudaFree(pixels);
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
      // this->removeSeam();
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
