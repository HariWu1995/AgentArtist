from PIL import Image

import math
import numpy as np
import torch
import torch.nn.functional as F


def save_img(img, output_path: str):
    img = (img.data.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(output_path)



