#include "lib.h"

#ifdef __cplusplus
extern "C"
{
#endif

  Carver *Carver_new(uint32_t *data, uint32_t width, uint32_t height);
  Carver *Carver_new_mask(uint32_t *data, uint8_t *mask, uint32_t width, uint32_t height);
  void Carver_delete(Carver *carver);
  int Carver_width(Carver *carver);
  int Carver_height(Carver *carver);
  void Carver_carve(Carver *carver, int numSeams);
  void Carver_getData(Carver *carver, uint32_t *dataptr);

#ifdef __cplusplus
}
#endif