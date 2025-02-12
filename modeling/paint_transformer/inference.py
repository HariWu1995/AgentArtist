import os
from tqdm import tqdm
from glob import glob
from PIL import Image

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.paint_transformer.network import Painter, SignWithSigmoidGrad as Decider
from modeling.paint_transformer.rendering import render_sequential as rendering
from modeling.paint_transformer.utilities import read_img, save_img, pad, crop, make_gif
from modeling.paint_transformer.brushstrokes import load_brushes


#################################
#       Global variables        #
#################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

PATCH_SIZE = 32
STROKE_NUM = 8
STROKE_PARAM = 5


#################################
#       Functionalities         #
#################################

def load_model(model_path: str):
    painter = Painter(param_per_stroke=STROKE_PARAM, total_strokes=STROKE_NUM, 
                            hidden_dim=256, n_heads=8, n_enc_layers=3, n_dec_layers=3)
    painter.load_state_dict(torch.load(model_path), strict=False)
    painter = painter.to(device=DEVICE).eval()
    for p in painter.parameters():
        p.requires_grad = False
    return painter


def run_pipeline(
        painter,
        brushes,
        image: Image.Image or str,
        verbose: bool = False,
        out_dir: str = './results',
    ):

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    original_img = read_img(image, 'RGB').to(device=DEVICE)
    original_h, original_w = original_img.shape[-2:]

    # Low-res -> Hi-res
    print('\nBaselining ...')

    K = max(math.ceil(math.log2(max(original_h, original_w) / PATCH_SIZE)), 0)

    original_img_pad_size = PATCH_SIZE * (2 ** K)
    original_img_pad = pad(original_img, original_img_pad_size, original_img_pad_size)

    final_result = torch.zeros_like(original_img_pad).to(device=DEVICE)

    pbar = tqdm(range(0, K+1))
    for layer in pbar:
        layer_size = PATCH_SIZE * (2 ** layer)
        patch_num = (layer_size - PATCH_SIZE) // PATCH_SIZE + 1

        pbar.set_description(f"layer {layer:02d} - size = {layer_size} - #patches = {patch_num}")

        img = F.interpolate(original_img_pad, (layer_size, layer_size))
        result = F.interpolate(final_result, (layer_size, layer_size))
        
        img_patch    = F.unfold(   img, (PATCH_SIZE, PATCH_SIZE), stride=(PATCH_SIZE, PATCH_SIZE))
        result_patch = F.unfold(result, (PATCH_SIZE, PATCH_SIZE), stride=(PATCH_SIZE, PATCH_SIZE))

        # img_patch, result_patch: b, 3 * output_size * output_size, h * w
        img_patch    =    img_patch.permute(0, 2, 1).contiguous().view(-1, 3, PATCH_SIZE, PATCH_SIZE).contiguous()
        result_patch = result_patch.permute(0, 2, 1).contiguous().view(-1, 3, PATCH_SIZE, PATCH_SIZE).contiguous()

        shape_param, \
        stroke_decision = painter(img_patch, result_patch)
        stroke_decision = Decider.apply(stroke_decision)

        grid = shape_param[:, :, :2].view(img_patch.shape[0] * STROKE_NUM, 1, 1, 2).contiguous()
        
        img_num = img_patch.shape[0]
        img_temp = img_patch.unsqueeze(1).contiguous().repeat(1, STROKE_NUM, 1, 1, 1)\
                                                .view(img_num * STROKE_NUM, 3, PATCH_SIZE, PATCH_SIZE).contiguous()

        color = F.grid_sample(img_temp, 2 * grid - 1, align_corners=False)\
                        .view(img_num, STROKE_NUM, 3).contiguous()

        stroke_param = torch.cat([shape_param, color], dim=-1)

        # stroke_param   : b * h * w, stroke_per_patch, param_per_stroke
        # stroke_decision: b * h * w, stroke_per_patch, 1
        param    =    stroke_param.view(1, patch_num, patch_num, STROKE_NUM, 8).contiguous()
        decision = stroke_decision.view(1, patch_num, patch_num, STROKE_NUM   ).contiguous().bool()

        # param   : b, h, w, stroke_per_patch, 8
        # decision: b, h, w, stroke_per_patch
        param[...,  :2] = param[...,  :2] / 2 + 0.25
        param[..., 2:4] = param[..., 2:4] / 2

        final_result = rendering(param, decision, brushes, final_result, 
                                out_dir, False, original_h, original_w, layer)

        if verbose:
            loss = ((final_result - original_img) ** 2).mean()
            print(f'L2Loss = {loss:.5f}')

    # Final refining
    print('\nRefining ...')

    border_size = original_img_pad_size // (2 * patch_num)
    layer_size = PATCH_SIZE * (2 ** layer)
    pad_size = PATCH_SIZE // 2

    img    = F.interpolate(original_img_pad, (layer_size, layer_size))
    result = F.interpolate(    final_result, (layer_size, layer_size))

    img    = F.pad(   img, [pad_size, pad_size, pad_size, pad_size, 0, 0, 0, 0])
    result = F.pad(result, [pad_size, pad_size, pad_size, pad_size, 0, 0, 0, 0])

    img_patch    = F.unfold(   img, (PATCH_SIZE, PATCH_SIZE), stride=(PATCH_SIZE, PATCH_SIZE))
    result_patch = F.unfold(result, (PATCH_SIZE, PATCH_SIZE), stride=(PATCH_SIZE, PATCH_SIZE))

    final_result = F.pad(final_result, [border_size, border_size, border_size, border_size, 0, 0, 0, 0])

    h = (img.shape[2] - PATCH_SIZE) // PATCH_SIZE + 1
    w = (img.shape[3] - PATCH_SIZE) // PATCH_SIZE + 1

    # img_patch, result_patch: b, 3 * output_size * output_size, h * w
    img_patch    =    img_patch.permute(0, 2, 1).contiguous().view(-1, 3, PATCH_SIZE, PATCH_SIZE).contiguous()
    result_patch = result_patch.permute(0, 2, 1).contiguous().view(-1, 3, PATCH_SIZE, PATCH_SIZE).contiguous()
    
    shape_param, stroke_decision = painter(img_patch, result_patch)

    grid = shape_param[:, :, :2].view(img_patch.shape[0] * STROKE_NUM, 1, 1, 2).contiguous()

    img_num = img_patch.shape[0]
    img_temp = img_patch.unsqueeze(1).contiguous().repeat(1, STROKE_NUM, 1, 1, 1)\
                                            .view(img_num * STROKE_NUM, 3, PATCH_SIZE, PATCH_SIZE).contiguous()
    
    color = F.grid_sample(img_temp, 2 * grid - 1, align_corners=False)\
                    .view(img_num, STROKE_NUM, 3).contiguous()
    
    stroke_param = torch.cat([shape_param, color], dim=-1)
    
    # stroke_param   : b * h * w, stroke_per_patch, param_per_stroke
    # stroke_decision: b * h * w, stroke_per_patch, 1
    param    =    stroke_param.view(1, h, w, STROKE_NUM, 8).contiguous()
    decision = stroke_decision.view(1, h, w, STROKE_NUM   ).contiguous() > 0 # https://github.com/Huage001/PaintTransformer/issues/10

    # param   : b, h, w, stroke_per_patch, 8
    # decision: b, h, w, stroke_per_patch
    param[...,  :2] = param[...,  :2] / 2 + 0.25
    param[..., 2:4] = param[..., 2:4] / 2

    final_result = rendering(param, decision, brushes, final_result, 
                            out_dir, True, original_h, original_w, layer+1)
    
    final_result = final_result[:, :, border_size:-border_size, border_size:-border_size]
    final_result = crop(final_result, original_h, original_w)

    final_path = os.path.join(out_dir, 'final.jpg')
    save_img(final_result[0], final_path)

    if verbose:
        loss = ((final_result - original_img) ** 2).mean()
        print(f'L2Loss = {loss:.5f}')


if __name__ == '__main__':

    # Load model & artefacts
    painter_ckpt_path = './checkpoints/paint_transformer/paint_best.pth'
    painter = load_model(painter_ckpt_path)
    brushes = load_brushes('brush_large').to(device=DEVICE)

    # Load image
    image_path = "C:/Users/Mr. RIAH/Pictures/_character/Nancy-Closeup.jpg"
    # image_path = "./samples/van-gogh-garden-at-arles.png"
    image_size = 1024
    image = Image.open(image_path).convert('RGB')
    # i_temp = Image.new(image.mode, (852, 852), (0, 0, 0)) 
    # i_temp.paste(image, (0, (852-552)//2))
    # image = i_temp
    image = image.resize((image_size, image_size))

    # Run pipeline
    out_dir = f'./results/nancy_{image_size}'
    with torch.no_grad():
        run_pipeline(painter, brushes, image, verbose=True, out_dir=out_dir)

    # Animation
    print('\nMaking GIF ...')
    images_list = [Image.open(f) for f in glob(f'{out_dir}/*.jpg')]
    make_gif(images_list, out_path=f'{out_dir}/out.gif')


