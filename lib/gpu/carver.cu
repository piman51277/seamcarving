#define __ARCH_GPU__
#include "../lib.h"

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

__device__ uint32_t distance(Color a, Color b)
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
          diffSum += distance((Color)tex2D<uint32_t>(pixels, i, j), (Color)tex2D<uint32_t>(pixels, i + k, j + l));
        }
      }
    }
    surf2Dwrite((uint32_t)(diffSum / hits), gradient, i * sizeof(uint32_t), j);
  }
}

__global__ void dpInitKernel(cudaTextureObject_t gradient, cudaSurfaceObject_t buf, uint32_t width, uint32_t height, uint32_t opwidth)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // compute the first row of the seam
  for (uint64_t i = index; i < opwidth; i += stride)
  {
    surf2Dwrite(tex2D<uint32_t>(gradient, i, 0), buf, i * sizeof(uint32_t), 0);
  }
}

__global__ void dpKernel(cudaTextureObject_t gradient, cudaSurfaceObject_t buf, uint32_t width, uint32_t row, uint32_t opwidth)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (uint64_t i = index; i < opwidth; i += stride)
  {
    uint64_t j = row;

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
  }
}

__global__ void findSeamKernel(cudaSurfaceObject_t buf, int *seam, uint32_t width, uint32_t height, uint32_t opwidth)
{

  // this should always execute within one thread block
  uint32_t ind = threadIdx.x;
  uint32_t stride = blockDim.x;
  __shared__ uint32_t minIndex[512];
  __shared__ uint32_t minVal[512];

  // find the minimum value in the last row
  uint32_t locMin = 0;
  uint32_t locMinVal = tex2D<uint32_t>(buf, 0, height - 1);
  for (uint64_t i = 1; i < opwidth; i += stride)
  {
    uint32_t val = tex2D<uint32_t>(buf, i, height - 1);
    if (val < locMinVal)
    {
      locMinVal = val;
      locMin = i;
    }
  }
  minIndex[ind] = locMin;
  minVal[ind] = locMinVal;

  __syncthreads();

  // parallel reduction to find the minimum value in the last row
  for (uint32_t s = stride / 2; s > 0; s >>= 1)
  {
    if (ind < s)
    {
      if (minVal[ind + s] < minVal[ind])
      {
        minVal[ind] = minVal[ind + s];
        minIndex[ind] = minIndex[ind + s];
      }
    }
    __syncthreads();
  }

  // from now on, it's a single thread
  if (ind != 0)
    return;
  seam[height - 1] = minIndex[0];

  // backtrack to find the seam
  uint32_t searchIndex = seam[height - 1];
  for (int64_t row = height - 2; row >= 0; row--)
  {
    uint64_t searchMin = searchIndex == 0 ? 0 : searchIndex - 1;
    uint64_t searchMax = searchIndex == opwidth - 1 ? opwidth - 1 : searchIndex + 1;

    uint32_t minVal = tex2D<uint32_t>(buf, searchIndex, row);
    for (uint64_t i = searchMin; i <= searchMax; i++)
    {
      uint32_t val = tex2D<uint32_t>(buf, i, row);
      if (val < minVal)
      {
        minVal = val;
        searchIndex = i;
      }
    }

    seam[row] = searchIndex;
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

  dpInitKernel<<<16, 256>>>(grad, buf, this->initialWidth, this->initialHeight, this->currentWidth);
  for (uint32_t i = 1; i < this->initialHeight; i++)
  {
    dpKernel<<<256, 32>>>(grad, buf, this->initialWidth, i, this->currentWidth);
  }
  cudaDeviceSynchronize();

  findSeamKernel<<<1, 256, (256 + 256) * sizeof(uint32_t)>>>(buf, this->seam, this->initialWidth, this->initialHeight, this->currentWidth);
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

void Carver::getGradient(uint32_t *outbuf)
{
  auto pix = createSurface(this->buf);
  uint32_t *pixels;
  cudaMallocManaged(&pixels, this->currentWidth * this->initialHeight * sizeof(uint32_t));
  copySurfaceArrKernel<<<64, 64>>>(pix, pixels, this->currentWidth, this->initialHeight, this->currentWidth);
  cudaDestroyTextureObject(pix);
  cudaDeviceSynchronize();
  std::copy(pixels, pixels + this->currentWidth * this->initialHeight, outbuf);
  cudaFree(pixels);
}

void Carver::getPixels(uint32_t *outbuf)
{
  auto pix = createSurface(this->pixels);
  uint32_t *pixels;
  cudaMallocManaged(&pixels, this->currentWidth * this->initialHeight * sizeof(uint32_t));
  copySurfaceArrKernel<<<64, 64>>>(pix, pixels, this->currentWidth, this->initialHeight, this->currentWidth);
  cudaDestroyTextureObject(pix);
  cudaDeviceSynchronize();
  std::copy(pixels, pixels + this->currentWidth * this->initialHeight, outbuf);
  cudaFree(pixels);
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

  for (uint32_t i = 0; i < count; i++)
  {
    this->computeGradient();
    this->computeSeam();
    this->removeSeam();
  }
}
