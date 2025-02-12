from PIL import Image

import math
import numpy as np
import torch
import torch.nn.functional as F


def read_img(img_path, img_type: str = 'RGB', h: int = None, w: int = None):
    if not isinstance(img_path, Image.Image):
        img = Image.open(img_path).convert(img_type)
    else:
        img = img_path
    if isinstance(h, int) \
    and isinstance(w, int):
        img = img.resize((w, h), resample=Image.NEAREST)
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float() / 255.
    return img


def save_img(img, output_path: str):
    img = (img.data.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(output_path)


def make_gif(all_frames, out_path: str, fps: int = 20, size: int = 512):
    all_frames = [f.resize((size, size)) for f in all_frames]
    init_frame = all_frames[0]
    init_frame.save(out_path, format="GIF", append_images=all_frames,
                                loop=0, fps=fps, save_all=True)


def pad(img, H, W):
    device = img.device
    b, c, h, w = img.shape
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    remainder_h = (H - h) % 2
    remainder_w = (W - w) % 2
    img = torch.cat([torch.zeros((b, c, pad_h, w), device=device), img,
                     torch.zeros((b, c, pad_h + remainder_h, w), device=device)], dim=-2)
    img = torch.cat([torch.zeros((b, c, H, pad_w), device=device), img,
                     torch.zeros((b, c, H, pad_w + remainder_w), device=device)], dim=-1)
    return img


def crop(img, h, w):
    H, W = img.shape[-2:]
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    remainder_h = (H - h) % 2
    remainder_w = (W - w) % 2
    img = img[:, :, pad_h : H - pad_h - remainder_h, 
                    pad_w : W - pad_w - remainder_w]
    return img



