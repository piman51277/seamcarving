#define __ARCH_CPU__
#include "../lib.h"

#include <iostream>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

namespace SeamCarver
{
  uint32_t distance(Color a, Color b)
  {
    uint8_t rd = abs(a.argb.r - b.argb.r);
    uint8_t gd = abs(a.argb.g - b.argb.g);
    uint8_t bd = abs(a.argb.b - b.argb.b);

    return fsqrt(rd * rd + gd * gd + bd * bd);
  }

  Carver::Carver(uint32_t *pixels, u_int8_t *mask, uint32_t width, uint32_t height)
  {
    this->pixels = new uint32_t[width * height];
    std::copy(pixels, pixels + width * height, this->pixels);
    this->mask = new uint8_t[width * height];
    std::copy(mask, mask + width * height, this->mask);
    this->initialWidth = width;
    this->initialHeight = height;
    this->currentWidth = width;
    this->gradient = new uint32_t[width * height];
    std::fill(this->gradient, this->gradient + width * height, 0);
    this->seam = new int[height];
    std::fill(this->seam, this->seam + height, 0);
    this->buf = new uint32_t[width * height];
    std::fill(this->buf, this->buf + width * height, 0x00FFFFFF);
  }

  Carver::Carver(uint32_t *pixels, uint32_t width, uint32_t height)
  {
    this->pixels = new uint32_t[width * height];
    std::copy(pixels, pixels + width * height, this->pixels);
    this->mask = new uint8_t[width * height];
    std::fill(this->mask, this->mask + width * height, 0);
    this->initialWidth = width;
    this->initialHeight = height;
    this->currentWidth = width;
    this->gradient = new uint32_t[width * height];
    std::fill(this->gradient, this->gradient + width * height, 0);
    this->seam = new int[height];
    std::fill(this->seam, this->seam + height, 0);
    this->buf = new uint32_t[width * height];
    std::fill(this->buf, this->buf + width * height, 0);
  }

  Carver::~Carver()
  {
    delete[] this->pixels;
    delete[] this->mask;
    delete[] this->gradient;
    delete[] this->seam;
    delete[] this->buf;
  }

  void Carver::computeGradient()
  {
    static bool first = true;

    if (first)
    {
      for (uint64_t j = 0; j < this->initialHeight; j++)
      {
        for (uint64_t i = 0; i < this->initialWidth; i++)
        {

          // 3x3 kernel
          float diffSum = 0;
          float hits = 0;
          for (int k = -1; k <= 1; k++)
          {
            for (int l = -1; l <= 1; l++)
            {
              if (i + k >= 0 && i + k < this->initialWidth && j + l >= 0 && j + l < this->initialHeight)
              {
                hits += 1;
                diffSum += distance((Color)this->pixels[j * this->initialWidth + i], (Color)this->pixels[(j + l) * this->initialWidth + i + k]);
              }
            }
          }
          this->gradient[j * this->initialWidth + i] = (uint32_t)(diffSum / hits * 150.0);
        }
      }
      first = false;
    }
    else
    {
      for (uint64_t j = 0; j < this->initialHeight; j++)
      {
        // get the seam at this row
        uint32_t seamIndex = this->seam[j];
        uint32_t minIndex = seamIndex <= 5 ? 0 : seamIndex - 5;
        uint32_t maxIndex = seamIndex + 5 >= this->currentWidth ? this->currentWidth : seamIndex + 5;

        for (uint64_t i = minIndex; i < maxIndex; i++)
        {

          // 3x3 kernel
          float diffSum = 0;
          float hits = 0;
          for (int k = -1; k <= 1; k++)
          {
            for (int l = -1; l <= 1; l++)
            {
              if (i + k >= 0 && i + k < this->currentWidth && j + l >= 0 && j + l < this->initialHeight)
              {
                hits += 1;
                diffSum += distance((Color)this->pixels[j * this->initialWidth + i], (Color)this->pixels[(j + l) * this->initialWidth + i + k]);
              }
            }
          }
          this->gradient[j * this->initialWidth + i] = (uint32_t)(diffSum / hits * 150.0);
        }
      }
    }
  }

  void Carver::computeSeam()
  {
    // compute the first row of the seam
    for (uint32_t i = 0; i < this->currentWidth; i++)
    {
      this->buf[i] = this->gradient[i];
    }

    // top-down DP approach
    for (uint64_t i = 1; i < this->initialHeight; i++)
    {
      for (uint64_t j = 0; j < this->currentWidth; j++)
      {
        uint64_t index = i * this->initialWidth + j;
        uint64_t prevRow = (i - 1) * this->initialWidth + j;

        if (j == 0)
        {
          this->buf[index] = this->gradient[index] + std::min(this->buf[prevRow], this->buf[prevRow + 1]);
        }
        else if (j == this->currentWidth - 1)
        {
          this->buf[index] = this->gradient[index] + std::min(this->buf[prevRow - 1], this->buf[prevRow]);
        }
        else
        {
          this->buf[index] = this->gradient[index] + std::min({this->buf[prevRow - 1], this->buf[prevRow], this->buf[prevRow + 1]});
        }
      }
    }

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
      uint32_t searchMin = minIndex == 0 ? 0 : minIndex - 1;
      uint32_t searchMax = minIndex == this->currentWidth - 1 ? this->currentWidth - 1 : minIndex + 1;

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
    for (uint32_t i = 0; i < this->initialHeight; i++)
    {
      uint64_t offset = i * this->initialWidth + this->seam[i];
      uint64_t maskOffset = this->currentWidth - this->seam[i] - 1;
      memcpy(this->pixels + offset, this->pixels + offset + 1, maskOffset * sizeof(uint32_t));
      memcpy(this->gradient + offset, this->gradient + offset + 1, maskOffset * sizeof(uint32_t));
      memcpy(this->mask + offset, this->mask + offset + 1, maskOffset * sizeof(uint8_t));
    }
    this->currentWidth--;
  }

  std::unique_ptr<Image> Carver::getGradient()
  {
    std::unique_ptr<Image> img(new Image(this->currentWidth, this->initialHeight));

    for (uint32_t i = 0; i < initialHeight; i++)
    {
      int offset = i * this->initialWidth;
      std::copy(this->gradient + offset, this->gradient + offset + this->currentWidth, img->pixels + i * this->currentWidth);
    }

    return img;
  }

  std::unique_ptr<Image> Carver::getPixels()
  {
    std::unique_ptr<Image> img(new Image(this->currentWidth, this->initialHeight));
    for (uint32_t i = 0; i < initialHeight; i++)
    {
      int offset = i * this->initialWidth;
      std::copy(this->pixels + offset, this->pixels + offset + this->currentWidth, img->pixels + i * this->currentWidth);
    }

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

      std::cout << "Gradient: " << duration1 << "us, Seam: " << duration2 << "us, Remove: " << duration3 << "us" << std::endl;
    }
  }
}
