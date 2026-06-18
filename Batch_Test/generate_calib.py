import numpy as np
from PIL import Image
from pathlib import Path

def preprocess(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    resize_shorter_side = 256
    input_w = input_h = 224

    if h < w:
        new_h = resize_shorter_side
        new_w = int(round(w * resize_shorter_side / h))
    else:
        new_w = resize_shorter_side
        new_h = int(round(h * resize_shorter_side / w))

    img = img.resize((new_w, new_h), resample=Image.BILINEAR)

    left = (new_w - input_w) // 2
    top = (new_h - input_h) // 2
    right = left + input_w
    bottom = top + input_h

    img = img.crop((left, top, right, bottom))

    arr = np.array(img).astype(np.float32) / 255.0
    return arr

paths = sorted(Path("./imagenet_calibration_data/dummy").rglob("*.JPEG"))[:1000]
data = np.stack([preprocess(p) for p in paths])  # [1000, 224, 224, 3]
np.save("calib_data.npy", data)