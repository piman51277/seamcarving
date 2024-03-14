from flask import Flask, request
from ctypes import *
from carver import Carver
import numpy as np
import time

app = Flask(__name__)

# set CORS to allow all


@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@app.route('/api/carve', methods=['POST'])
def carve_handler():

    # extract the binary data from the request
    data: bytes = request.get_data(cache=False)

    # read the first 4 bytes to determine how many seams to carve
    num_seams = int.from_bytes(data[:4], byteorder='little')

    (width, height, pixel_buf, mask_buf) = parse_image(data[4:])

    # just a quick sanity check... the number of seams should be less than the width
    if num_seams >= width:
        return "Invalid number of seams", 400

    # create a Carver object
    carver = Carver(pixel_buf, width, height)
    carver.carve(num_seams)
    resultBuf = carver.getData()

    carver.delete()

    # write the result back to the client
    return write_image(width - num_seams, height, resultBuf)


def parse_image(data: bytes) -> (int, int, np.ndarray, np.ndarray):  # type: ignore
    # read the first 4 bytes at uint32 LE
    width = int.from_bytes(data[:4], byteorder='little')
    height = int.from_bytes(data[4:8], byteorder='little')

    # next byte will tell us if we have a mask
    mask_present = data[8] == 1

    pixel_buf_len = width * height * 4
    pixel_buf = np.frombuffer(data[9:9 + pixel_buf_len], dtype=np.uint32)

    mask_buf = None
    if mask_present:
        mask_buf_len = width * height
        mask_buf = np.frombuffer(
            data[9 + pixel_buf_len:9 + pixel_buf_len + mask_buf_len], dtype=np.uint8)

    return width, height, pixel_buf, mask_buf


def write_image(width, height, pixel_buf, mask_buf=None):
    buf = bytes()
    buf += width.to_bytes(4, byteorder='little')
    buf += height.to_bytes(4, byteorder='little')
    buf += (1 if mask_buf is not None else 0).to_bytes(1, byteorder='little')
    buf += pixel_buf.tobytes()
    if mask_buf is not None:
        buf += mask_buf.tobytes()
    return buf
