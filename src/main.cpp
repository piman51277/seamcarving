#include "lib/lib.h"
#include <iostream>
#include <png.h>
#include <stdint.h>
#include <stdlib.h>
#include <SDL2/SDL.h>

uint8_t *read_png_file(const char *file_name, int *width, int *height)
{
  FILE *fp = fopen(file_name, "rb");

  png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  png_infop info = png_create_info_struct(png);

  png_init_io(png, fp);
  png_read_info(png, info);

  *width = png_get_image_width(png, info);
  *height = png_get_image_height(png, info);

  png_read_update_info(png, info);

  uint8_t *image_data = new uint8_t[*width * *height * 3];
  png_bytep *row_pointers = new png_bytep[*height];

  for (int y = 0; y < *height; y++)
  {
    row_pointers[y] = (png_byte *)&image_data[y * *width * 3];
  }

  png_read_image(png, row_pointers);

  fclose(fp);
  png_destroy_read_struct(&png, &info, NULL);
  delete[] row_pointers;

  return image_data;
}

int main()
{
  int width, height;
  uint8_t *pixels = read_png_file("image.png", &width, &height);
  SDL_Init(SDL_INIT_EVERYTHING);
  SDL_Window *window = SDL_CreateWindow("Seam Carving", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, 0);
  SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
  SDL_RenderSetLogicalSize(renderer, width, height);

  uint32_t *pixels32 = new uint32_t[width * height];
  for (int i = 0; i < width * height; i++)
  {
    pixels32[i] = (pixels[i * 3] << 16) | (pixels[i * 3 + 1] << 8) | pixels[i * 3 + 2];
    // set alpha to 255
    pixels32[i] |= 0xFF000000;
  }

  SeamCarver::Carver carver(pixels32, width, height);

  while (true)
  {
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
      if (event.type == SDL_QUIT)
      {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 0;
      }
    }

    carver.removeSeams(1);
    std::shared_ptr<SeamCarver::Image> gradient = carver.getPixels();
    uint32_t *gradientPixels = gradient->pixels;
    uint32_t Iwidth = gradient->width;

    // set background to black
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);

    // create a texture from the pixel array
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STATIC, Iwidth, height);
    SDL_UpdateTexture(texture, NULL, gradientPixels, Iwidth * sizeof(uint32_t));
    SDL_Rect *dstrect = new SDL_Rect();
    dstrect->x = 0;
    dstrect->y = 0;
    dstrect->w = Iwidth;
    dstrect->h = height;
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, NULL, dstrect);
    SDL_RenderPresent(renderer);

    // cleanup
    delete dstrect;
    SDL_DestroyTexture(texture);

    SDL_Delay(1);
  }
}