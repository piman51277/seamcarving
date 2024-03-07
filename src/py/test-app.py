import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from ctypes import *
import requests

# URL of the server
url = "http://localhost:5000/api/carve"
num_remove = 100

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

# construct packet
pktbuf = bytearray()
pktbuf += num_remove.to_bytes(4, byteorder='little')
pktbuf += width.to_bytes(4, byteorder='little')
pktbuf += height.to_bytes(4, byteorder='little')
pktbuf += 0x00.to_bytes(1, byteorder='little')  # no mask
pktbuf += data.tobytes()

# send the packet
response = requests.post(url, data=pktbuf)

# get the binary body of the response
rspdata = response.content

new_width = int.from_bytes(rspdata[:4], byteorder='little')
new_height = int.from_bytes(rspdata[4:8], byteorder='little')
new_pix = np.frombuffer(rspdata[9:], dtype=np.uint32)

new_pix = new_pix.reshape(new_height, new_width)

# convert out back into rgb
r = (new_pix >> 16) & 0xff
g = (new_pix >> 8) & 0xff
b = new_pix & 0xff

short_img = np.dstack((r, g, b))

plt.imshow(short_img)
plt.show()
