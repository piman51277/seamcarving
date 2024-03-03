#define __ARCH_CPU__
#include "../lib.h"

#include <iostream>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

namespace SeamCarver
{
  float _toLinear(float c)
  {
    if (c <= 0.04045)
    {
      return c / 12.92;
    }
    else
    {
      return pow((c + 0.055) / 1.055, 2.4);
    }
  }

  CIELAB rgb2lab(Color c)
  {
    float r = _toLinear(c.argb.r / 255.0);
    float g = _toLinear(c.argb.g / 255.0);
    float b = _toLinear(c.argb.b / 255.0);

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

    CIELAB lab;
    lab.l = 116.0 * y - 16.0;
    lab.a = 500.0 * (x - y);
    lab.b = 200.0 * (y - z);

    return lab;
  }

  float distance(CIELAB a, CIELAB b)
  {
    float dl = a.l - b.l;
    float da = a.a - b.a;
    float db = a.b - b.b;

    return sqrt(dl * dl + da * da + db * db);
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
    this->lab = new CIELAB[width * height];
    std::fill(this->lab, this->lab + width * height, CIELAB{0, 0, 0});
    this->gradient = new uint32_t[width * height];
    std::fill(this->gradient, this->gradient + width * height, 0);
    this->seam = new int[height];
    std::fill(this->seam, this->seam + height, 0);
    this->buf = new uint32_t[width * height];
    std::fill(this->buf, this->buf + width * height, 0x00FFFFFF);

    // convert the pixels to CIELAB
    for (uint64_t i = 0; i < width * height; i++)
    {
      this->lab[i] = rgb2lab({.value = pixels[i]});
    }
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
    this->lab = new CIELAB[width * height];
    std::fill(this->lab, this->lab + width * height, CIELAB{0, 0, 0});
    this->gradient = new uint32_t[width * height];
    std::fill(this->gradient, this->gradient + width * height, 0);
    this->seam = new int[height];
    std::fill(this->seam, this->seam + height, 0);
    this->buf = new uint32_t[width * height];
    std::fill(this->buf, this->buf + width * height, 0);

    // convert the pixels to CIELAB
    for (uint64_t i = 0; i < width * height; i++)
    {
      this->lab[i] = rgb2lab({.value = pixels[i]});
    }
  }

  Carver::~Carver()
  {
    delete[] this->pixels;
    delete[] this->lab;
    delete[] this->mask;
    delete[] this->gradient;
    delete[] this->seam;
    delete[] this->buf;
  }

  void Carver::computeGradient()
  {
    for (uint64_t i = 0; i < this->currentWidth; i++)
    {
      for (uint64_t j = 0; j < this->initialHeight; j++)
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
              diffSum += distance(this->lab[j * this->initialWidth + i], this->lab[(j + l) * this->initialWidth + i + k]);
            }
          }
        }
        this->gradient[j * this->initialWidth + i] = (uint32_t)(diffSum / hits * 150.0);
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
      memcpy(this->lab + offset, this->lab + offset + 1, maskOffset * sizeof(CIELAB));
      memcpy(this->gradient + offset, this->gradient + offset + 1, maskOffset * sizeof(uint32_t));
      memcpy(this->buf + offset, this->buf + offset + 1, maskOffset * sizeof(uint32_t));
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
