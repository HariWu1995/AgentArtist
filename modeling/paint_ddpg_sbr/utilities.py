from pathlib import Path
import tempfile

import cv2
import imageio
from PIL import Image

import numpy as np
import torch


def small2large(x: np.ndarray, width: int, division: int):
    # (d * d, width, width) -> (d * width, d * width)    
    x = x.reshape(division, division, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(division * width, division * width, -1)
    return x


def large2small(x: np.ndarray, num_canvas: int, width: int, division: int):
    # (d * width, d * width) -> (d * d, width, width)
    x = x.reshape(division, width, division, width, 3)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(num_canvas, width, width, 3)
    return x


def smoothing(img: np.ndarray, width: int, division: int):

    def smooth_pix(img, tx, ty):
        if (tx == division * width - 1) \
        or (ty == division * width - 1) \
        or (tx == 0) \
        or (ty == 0): 
            return img
        img[tx, ty] = (img[tx, ty] + img[tx + 1, ty    ] + img[tx    , ty + 1] + \
                                     img[tx - 1, ty    ] + img[tx    , ty - 1] + \
                                     img[tx + 1, ty - 1] + img[tx - 1, ty + 1] + \
                                     img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
        return img

    for p in range(division):
        for q in range(division):
            x = p * width
            y = q * width
            for k in range(width):
                img = smooth_pix(img, x + k, y + width - 1)
                if q != division - 1:
                    img = smooth_pix(img, x + k, y + width)
            for k in range(width):
                img = smooth_pix(img, x + width - 1, y + k)
                if p != division - 1:
                    img = smooth_pix(img, x + width, y + k)
    return img


def preprocess(image: np.ndarray, num_canvas: int, width: int, division: int, 
                hires: bool = False, device = None):
    if hires:
        return preprocess_hires(image, num_canvas, width, division, device)
    else:
        return preprocess_lowres(image, num_canvas, width, division, device)

    
def preprocess_hires(image: np.ndarray, num_canvas: int, width: int, division: int, device = None):

    # divide target to patches (if division > 1)
    patch_size = tuple([width * division] * 2)
    patch_img = cv2.resize(image, patch_size)
    patch_img = large2small(patch_img, num_canvas, width, division)
    patch_img = np.transpose(patch_img, (0, 3, 1, 2))
    patch_img = torch.tensor(patch_img).float()

    if device is not None:
        patch_img = patch_img.to(device)

    return patch_img

    
def preprocess_lowres(image: np.ndarray, num_canvas: int, width: int, division: int, device = None):

    # divide target to patches (if division > 1)
    patch_size = tuple([width * division] * 2)
    patch_img = cv2.resize(image, patch_size)
    patch_img = large2small(patch_img, num_canvas, width, division)
    patch_img = np.transpose(patch_img, (0, 3, 1, 2))
    patch_img = torch.tensor(patch_img).float() / 255.

    # resize image to feed into model
    sized_img = cv2.resize(image, (width, width))
    sized_img = sized_img.reshape(1, width, width, 3)
    sized_img = np.transpose(sized_img, (0, 3, 1, 2))
    sized_img = torch.tensor(sized_img).float() / 255.

    if device is not None:
        patch_img = patch_img.to(device)
        sized_img = sized_img.to(device)

    return patch_img, sized_img


def postprocess(output, width: int = -1, division: int = -1, out_size = None, is_divided: bool = False):
    output = output.detach().cpu().numpy() # d * d, 3, width, width    
    output = np.transpose(output, (0, 2, 3, 1))
    if is_divided:
        assert (width > 0) and (division > 0), \
            f"(width = {width}) & (division = {division}) must be positive integers."
        output = small2large(output, width, division)
        output = smoothing(output, width, division)
    else:
        output = output[0]
    output = (output * 255).astype('uint8')
    if isinstance(out_size, (list, tuple)):
        output = cv2.resize(output, out_size)
    # output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def make_gif(all_frames, out_path: str, fps: int = 20, size: int = 512):
    # imageio.mimwrite(out_path, seq_frames, duration=0.02)
    seq_frames = []
    for f in all_frames:
        f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        f = Image.fromarray(f).resize((size, size))
        seq_frames.append(f)
    init_frame = seq_frames[0]
    init_frame.save(out_path, format="GIF", append_images=seq_frames,
                                loop=0, fps=fps, save_all=True)

