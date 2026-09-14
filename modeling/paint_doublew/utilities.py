from PIL import Image

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .compositor.Renderer import morphology
from .compositor.Renderer.stroke_gen import draw_oil


transform_img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([512, 512]),
])

resize_64  = transforms.Resize(( 64,  64))
resize_128 = transforms.Resize((128, 128))
resize_512 = transforms.Resize((512, 512))
resize_256 = transforms.Resize((256, 256))

WIDTH = 128 * 4
WIDTH_OUT = 512
PARAM_NUM = 5


def small2large(x, width, K):
    # (d * d, width, width) -> (d * width, d * width)
    x = x.reshape(K, K, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(K * width, K * width, -1)
    return x


def large2small(x, width, K, C):
    # (d * width, d * width) -> (d * d, width, width)
    x = x.reshape(K, width, K, width, 3)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(C, width, width, 3)
    return x


def decode3(x, canvas, meta_brushes):  # b * (10 + 3)
    x = x.view(-1, PARAM_NUM + 3)
    tmp = 1 - draw_oil(meta_brushes, x[:, :PARAM_NUM])
    stroke = tmp[:, 0]
    alpha  = tmp[:, 1]
    stroke = stroke.view(-1, 128, 128, 1)
    alpha  =  alpha.view(-1, 128, 128, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    alpha        =        alpha.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    alpha        =        alpha.view(-1, 5, 1, 128, 128)
    color_stroke = color_stroke.view(-1, 5, 3, 128, 128)
    res = []
    for i in range(5):
        canvas = canvas * (1 - alpha[:, i]) + alpha[:, i] * color_stroke[:, i]
        res.append(canvas)
    return canvas, res


def decode_oil(meta_brushes, x, size=512):
    tmp = 1 - draw_oil(meta_brushes, x[:, :PARAM_NUM], size=size)
    stroke = tmp[:, 0]
    alpha  = tmp[:, 1]
    stroke = stroke.view(-1, size, size, 1)
    alpha  =  alpha.view(-1, size, size, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    alpha        =        alpha.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    alpha        =        alpha.view(-1, 5, 1, size, size)
    color_stroke = color_stroke.view(-1, 5, 3, size, size)
    return  color_stroke, alpha


def decode(
    Painter, 
    meta_brushes, 
    box, 
    params,
    canvas, 
    tgt_canvas, 
    debug=False,
):  
    # b * (10 + 3)
    ori_canvas = canvas.clone()
    canvas     = resize_128(canvas)
    tgt_canvas = resize_128(tgt_canvas)
    
    for i in range(canvas.size(0)):
        x1, y1, x2, y2 = torch.round(box[i] * 127).detach().int()
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        resize = transforms.Resize((4 * (x2 + 1 - x1), 
                                    4 * (y2 + 1 - y1)))

        tar_canvas_box = resize_128(tgt_canvas[i, :, x1:x2 + 1, y1:y2 + 1]).unsqueeze(0)

        for kk in range(1):
            canvas     = resize_128(ori_canvas)
            canvas_box = resize_128(canvas[i, :, x1:x2 + 1, y1:y2 + 1]).unsqueeze(0)
            
            param = Painter(torch.cat((canvas_box, tar_canvas_box), dim=1))
            params.append(param)
            x = param.view(-1, PARAM_NUM+3)
            
            # foregrounds, alphas, _ = Decoder(x[:, :PARAM_NUM+3])
            foregrounds, alphas = decode_oil(meta_brushes, x[:, :PARAM_NUM+3])
            foregrounds = foregrounds.view(-1, 5, 3, 512, 512)
            alphas      =      alphas.view(-1, 5, 1, 512, 512)
            for j in range(5):
                ori_canvas[i, :, 4*x1:4*(x2+1), 4*y1:4*y2+4] = \
                ori_canvas[i, :, 4*x1:4*(x2+1), 4*y1:4*y2+4] * resize(1 - alphas[0, j]) \
                                     + resize(alphas[0, j]) * resize(foregrounds[0, j])
    return ori_canvas, params


def final_decode(
    canvas,
    boxes0, 
    boxes1, 
    params, 
    meta_brushes,
    recursive_number: int,
):
    # canvas = torch.zeros(1, 3, WIDTH_OUT, WIDTH_OUT).cuda()
    images_list = []

    for i, box0 in enumerate(boxes0):
        x01, y01, x02, y02 = box0
        x01, y01, x02, y02 = min(x01, x02), min(y01, y02), \
                             max(x01, x02), max(y01, y02)        
        w0 = x02 - x01
        h0 = y02 - y01

        for j in range(recursive_number):
            x11, y11, x12, y12 = boxes1.pop(0)
            x11, y11, x12, y12 = min(x11, x12), min(y11, y12), max(x11, x12), max(y11, y12)
            x1 = int((x01 + x11 * w0) * (WIDTH_OUT - 1))
            x2 = int((x01 + x12 * w0) * (WIDTH_OUT - 1))
            y1 = int((y01 + y11 * h0) * (WIDTH_OUT - 1))
            y2 = int((y01 + y12 * h0) * (WIDTH_OUT - 1))
            
            resize = transforms.Resize((x2 + 1 - x1, y2 + 1 - y1))

            # for k in range(1):
            param = params.pop(0)
            x = param.view(-1, PARAM_NUM+3)

            foregrounds, alphas = decode_oil(meta_brushes, x[:, :PARAM_NUM+3])
            foregrounds = foregrounds.view(-1, 5, 3, 512, 512)
            alphas      =      alphas.view(-1, 5, 1, 512, 512)
            # foregrounds[0] = morphology.dilation(foregrounds[0])
            # alphas[0]      = morphology.erosion(alphas[0])

            for k in range(5):
                canvas[0, :, x1:(x2+1),  y1:y2+1] = \
                canvas[0, :, x1:(x2+1),  y1:y2+1] * resize(1 - alphas[0, k]) \
                          + resize(alphas[0, k]) * resize(foregrounds[0, k])

            frame = (canvas[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            images_list.append(frame)

    return canvas, images_list


def read_img(img_path, img_type: str = 'RGB', h: int = None, w: int = None, 
                    return_type: str = 'torch'):
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
    img = img.astype(np.float32) / 255.
    if return_type == 'torch':
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)
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


