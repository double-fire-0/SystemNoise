import io
from PIL import Image
import logging
import cv2
import ffmpeg
import numpy as np

logger = logging.getLogger('global')


def pil_loader(img_bytes, filepath):
    buff = io.BytesIO(img_bytes)
    try:
        with Image.open(buff) as img:
            img = img.convert('RGB')
    except IOError:
        logger.info('Failed in loading {}'.format(filepath))
    return img


def opencv_loader(img_bytes, filepath):
    try:
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        color_mode = 'RGB'
        if color_mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_mode == "GRAY":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    except IOError:
        logger.info('Failed in loading {}'.format(filepath))

def ffmpeg_loader(img_bytes, filepath):
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    height = img.shape[0]
    width = img.shape[1]
    out, _ = (
        ffmpeg
        .input(filepath)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True)
    )
    img = (
            np
            .frombuffer(out, np.uint8)
            .reshape([height, width, 3])
    )
    return img

def npy_loader(img_bytes, filepath):
    img = np.load(filepath).astype(np.uint8)
    return img



def build_image_reader(reader_type):
    if reader_type == 'pil':
        return pil_loader
    elif reader_type == 'opencv':
        return opencv_loader
    elif reader_type == 'ffmpeg':
        return ffmpeg_loader
    elif reader_type == 'npy':
        return npy_loader
    else:
        raise NotImplementedError
