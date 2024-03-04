#define __ARCH_CPU__
#include "lib.h"
#include "wrap.h"

extern "C"
{
  // these pointers are copied immediately so its fine!
  Carver *Carver_new(uint32_t *data, uint32_t width, uint32_t height)
  {
    return new Carver(data, width, height);
  }

  Carver *Carver_new_mask(uint32_t *data, uint8_t *mask, uint32_t width, uint32_t height)
  {
    return new Carver(data, mask, width, height);
  }

  void Carver_delete(Carver *carver)
  {
    delete carver;
  }

  int Carver_width(Carver *carver)
  {
    return carver->width();
  }

  int Carver_height(Carver *carver)
  {
    return carver->height();
  }

  void Carver_carve(Carver *carver, int numSeams)
  {
    carver->removeSeams(numSeams);
  }

  void Carver_getData(Carver *carver, uint32_t *dataptr)
  {
    carver->getPixels(dataptr);
  }
}