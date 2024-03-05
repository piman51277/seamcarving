import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from ctypes import *
from carver import Carver
# testing code

img = mpimg.imread('image.png')
plt.imshow(img)

# get the width and height of the image
width = img.shape[1]
height = img.shape[0]
depth = img.shape[2]

# flatten the image
img = img.flatten()

# for each x,y pixel, convert the RGB values to a single uint32
# format is 0xAARRGGBB
data = np.zeros(width * height, dtype=np.uint32)

# Scale the RGB values
img = (img * 255).astype(np.uint32)

# Combine the RGB values into a single uint32
data = (255 << 24) | (img[0::3] << 16) | (img[1::3] << 8) | img[2::3]

print("width: ", width)
print("height: ", height)
print("depth: ", depth)

# create a Carver object
carver = Carver(data, width, height)

# carve 400 seams
carver.carve(400)

newWidth = carver.width()
newHeight = carver.height()

print("newWidth: ", newWidth)
print("newHeight: ", newHeight)

# get the data from the carver
out = carver.getData()

# convert out back into rgb
r = (out >> 16) & 0xff
g = (out >> 8) & 0xff
b = out & 0xff

out = np.dstack((r, g, b))

plt.imshow(out)
plt.show()
