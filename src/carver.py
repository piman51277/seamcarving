import numpy as np
from ctypes import *

lib = cdll.LoadLibrary('./bin/libgpu.so')

# lib.Carver_new makes the following:
# uint32_t *data;
# uint32_t width;
# uint32_t height;
lib.Carver_new.argtypes = [POINTER(c_uint32), c_uint32, c_uint32]

# lib.Carver_new_mask makes the following:
# uint32_t *data;
# uint8_t *mask;
# uint32_t width;
# uint32_t height;
lib.Carver_new_mask.argtypes = [
    POINTER(c_uint32), POINTER(c_uint8), c_uint32, c_uint32]

# lib.Carver_delete takes the following:
# void *carver;
lib.Carver_delete.argtypes = [c_void_p]

# lib.Carver_width returns the width of the image
lib.Carver_width.argtypes = [c_void_p]
lib.Carver_width.restype = c_uint32

# lib.Carver_height returns the height of the image
lib.Carver_height.argtypes = [c_void_p]
lib.Carver_height.restype = c_uint32

# lib.Carver_carve takes the following:
# void *carver;
# uint32_t numSeams;
lib.Carver_carve.argtypes = [c_void_p, c_uint32]

# lib.Carver_getData takes the following:
# void *carver;
# uint32_t *out;
lib.Carver_getData.argtypes = [c_void_p, POINTER(c_uint32)]
lib.Carver_getData.restype = POINTER(c_uint32)


class Carver(object):
    def __init__(self, data: np.ndarray[np.uint32], width: int, height: int, mask: np.ndarray[np.uint8] = None):
        self.data = data
        if mask is not None:
            self.mask = mask
            self.carver = lib.Carver_new_mask(
                data.ctypes.data_as(POINTER(c_uint32)), mask.ctypes.data_as(POINTER(c_uint8)), width, height)
        else:
            self.carver = lib.Carver_new(
                data.ctypes.data_as(POINTER(c_uint32)), width, height)

    def __del__(self):
        lib.Carver_delete(self.carver)

    def carve(self, numSeams: int):
        lib.Carver_carve(self.carver, numSeams)

    def width(self) -> int:
        return lib.Carver_width(self.carver)

    def height(self) -> int:
        return lib.Carver_height(self.carver)

    def getData(self) -> np.ndarray[np.uint32]:
        out = np.zeros(self.width() * self.height(), dtype=np.uint32)
        lib.Carver_getData(self.carver, out.ctypes.data_as(POINTER(c_uint32)))
        return out.reshape(self.height(), self.width())
