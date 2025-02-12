import os
from tqdm import tqdm
from glob import glob
from PIL import Image

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.paint_transformer.inference_torch.network import Painter, SignWithSigmoidGrad as Decider
from modeling.paint_transformer.inference_torch.morphology import erosion, dilation
from modeling.paint_transformer.utilities import read_img, save_img, pad, crop, make_gif


#################################
#       Global variables        #
#################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

PATCH_SIZE = 32
STROKE_NUM = 8
STROKE_PARAM = 5

BRUSH_FOLDER = './checkpoints/paint_transformer/brush'
BRUSH_TYPE = 'brush_large'


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


def load_brushes(brush_type: str = BRUSH_TYPE):
    brush_v = read_img(f'{BRUSH_FOLDER}/{brush_type}_vertical.png', 'L').to(device=DEVICE)
    brush_h = read_img(f'{BRUSH_FOLDER}/{brush_type}_horizontal.png', 'L').to(device=DEVICE)
    brushes = torch.cat([brush_v, brush_h], dim=0)
    return brushes


def param2stroke(param, H, W, brushes):
    """
    Input a set of stroke parameters and output its corresponding foregrounds and alpha maps.

    Args:
        param: a tensor with shape n_strokes x n_param_per_stroke, 
                    where n_param_per_stroke = 8, including: 
                            x_center, y_center, width, height, theta, R, G, and B.
        H: output height.
        W: output width.
        brushes: a tensor with shape 2 x 3 x brush_height x brush_width.
                    On the batch dimension, the 1st slice denotes vertical brush and 
                                            the 2nd slice denotes horizontal brush.

    Returns:
        foregrounds: a tensor with shape n_strokes x 3 x H x W, containing color information.
             alphas: a tensor with shape n_strokes x 3 x H x W, containing binary information 
                    of whether a pixel is belonging to the stroke (alpha mat), for painting process.
    """
    # Firstly, resize the brushes to the required shape,
    # in order to decrease GPU memory especially when the required shape is small.
    brushes = F.interpolate(brushes, (H, W))
    b = param.shape[0]

    # Extract shape parameters and color parameters.
    param_list = torch.split(param, 1, dim=1)
    x0, y0, w, h, theta = [item.squeeze(-1) for item in param_list[:5]]
    R, G, B = param_list[5:]

    # Pre-compute sin theta and cos theta
    sin_theta = torch.sin(torch.acos(torch.tensor(-1., device=param.device)) * theta)
    cos_theta = torch.cos(torch.acos(torch.tensor(-1., device=param.device)) * theta)

    # index indicates each stroke should be in which direction (vertical or horizontal).
    # When h > w, vertical stroke should be used. 
    # When h <= w, horizontal stroke should be used.
    index = torch.full((b,), -1, device=param.device, dtype=torch.long)
    index[h > w] = 0
    index[h <= w] = 1
    brush = brushes[index.long()]

    # Calculate warp matrix according to the rules defined by pytorch, in order for warping.
    warp_00 = cos_theta / w
    warp_01 = sin_theta * H / (W * w)
    warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
    
    warp_10 = -sin_theta * W / (H * h)
    warp_11 = cos_theta / h
    warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
    
    warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
    warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
    warp = torch.stack([warp_0, warp_1], dim=1)

    # Conduct warping.
    grid = F.affine_grid(warp, [b, 3, H, W], align_corners=False)
    brush = F.grid_sample(brush, grid, align_corners=False)

    # alphas is the binary information suggesting whether a pixel is belonging to the stroke.
    alphas = (brush > 0).float()
    brush = brush.repeat(1, 3, 1, 1)
    alphas = alphas.repeat(1, 3, 1, 1)

    # Give color to foreground strokes.
    color_map = torch.cat([R, G, B], dim=1)
    color_map = color_map.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)
    foreground = brush * color_map

    # Dilation and erosion are used for foregrounds and alphas respectively to prevent artifacts on stroke borders.
    foreground = dilation(foreground)
    alphas = erosion(alphas)

    return foreground, alphas


def param2img_sequential(
        param, decision, brushes, cur_canvas, 
        frame_dir: str = None, has_border: bool = False, 
        original_h: int = None, original_w: int = None, 
        layer_id: int = 0,
    ):
    """
    Input stroke parameters and decisions for each patch, brushes, current canvas, frame directory,
    and whether there is a border (if intermediate painting results are required).

    Output the painting results of adding the corresponding strokes on the current canvas.

    Args:
        param: a tensor with shape 
                batch size x patch along height dimension x patch along width dimension x n_stroke_per_patch x n_param_per_stroke
        decision: a binary tensor with shape 
                batch size x patch along height dimension x patch along width dimension x n_stroke_per_patch
        brushes: a tensor with shape 2 x 3 x brush_height x brush_width.
                On the batch dimension, the 1st slice denotes vertical brush and 
                                        the 2nd one denotes horizontal brush.
        cur_canvas: a tensor with shape batch size x 3 x H x W,
                where H and W denote height and width of padded results of original images.
        frame_dir: directory to save intermediate painting results. 
                None means intermediate results are not required.
        has_border: on the last painting layer, in order to make sure that the painting results do not miss any important detail, 
                we choose to paint again on this layer but shift PATCH_SIZE // 2 pixels when cutting patches. 
                In this case, if intermediate results are required, we need to cut the shifted length on the border before saving, 
                or there would be a black border.
        original_h: to indicate the original height for cropping when saving intermediate results.
        original_w: to indicate the original width for cropping when saving intermediate results.

    Returns:
        cur_canvas: a tensor with shape batch size x 3 x H x W, denoting painting results.
    """
    # param   : b, h, w, num_strokes_per_patch, num_params_per_stroke
    # decision: b, h, w, num_strokes_per_patch
    b, h, w, s, p = param.shape
    H, W = cur_canvas.shape[-2:]
    device = cur_canvas.device

    is_odd_y = h % 2 == 1
    is_odd_x = w % 2 == 1

    patch_size_y = 2 * H // h
    patch_size_x = 2 * W // w

    factor = 2 if has_border else 4
    img_id = 0

    even_idx_y = torch.arange(0, h, 2, device=device)
    even_idx_x = torch.arange(0, w, 2, device=device)
    odd_idx_y = torch.arange(1, h, 2, device=device)
    odd_idx_x = torch.arange(1, w, 2, device=device)

    even_y_even_x_coord_y, even_y_even_x_coord_x = torch.meshgrid([even_idx_y, even_idx_x])
    even_y_odd_x_coord_y, even_y_odd_x_coord_x = torch.meshgrid([even_idx_y, odd_idx_x])
    odd_y_even_x_coord_y, odd_y_even_x_coord_x = torch.meshgrid([odd_idx_y, even_idx_x])
    odd_y_odd_x_coord_y, odd_y_odd_x_coord_x = torch.meshgrid([odd_idx_y, odd_idx_x])

    cur_canvas = F.pad(cur_canvas, [patch_size_x // 4, patch_size_x // 4,
                                    patch_size_y // 4, patch_size_y // 4, 0, 0, 0, 0])

    def partial_render(this_canvas, patch_coord_y, patch_coord_x, stroke_id):
        canvas_patch = F.unfold(this_canvas, (patch_size_y, patch_size_x),
                                stride=(patch_size_y // 2, patch_size_x // 2))

        # canvas_patch: b, 3 * py * px, h * w
        canvas_patch = canvas_patch.view(b, 3, patch_size_y, patch_size_x, h, w).contiguous()
        canvas_patch = canvas_patch.permute(0, 4, 5, 1, 2, 3).contiguous()

        # canvas_patch: b, h, w, 3, py, px
        selected_canvas_patch = canvas_patch[:, patch_coord_y, patch_coord_x, :, :, :]
        selected_h, selected_w = selected_canvas_patch.shape[1:3]
        
        selected_param    =    param[:, patch_coord_y, patch_coord_x, stroke_id, :].view(-1, p).contiguous()
        selected_decision = decision[:, patch_coord_y, patch_coord_x, stroke_id   ].view(-1   ).contiguous()

        selected_foregrounds = torch.zeros(selected_param.shape[0], 3, patch_size_y, patch_size_x, device=device)
        selected_alphas      = torch.zeros(selected_param.shape[0], 3, patch_size_y, patch_size_x, device=device)

        if selected_param[selected_decision, :].shape[0] > 0:
            selected_foregrounds[selected_decision, :, :, :], \
                 selected_alphas[selected_decision, :, :, :] = param2stroke(selected_param[selected_decision, :], 
                                                                            patch_size_y, patch_size_x, brushes)

        selected_foregrounds = selected_foregrounds.view(b, selected_h, selected_w, 3, patch_size_y, patch_size_x).contiguous()
        selected_alphas      =      selected_alphas.view(b, selected_h, selected_w, 3, patch_size_y, patch_size_x).contiguous()
        selected_decision    =    selected_decision.view(b, selected_h, selected_w, 1, 1, 1).contiguous()

        selected_canvas_patch = selected_foregrounds * selected_alphas * selected_decision + \
                                selected_canvas_patch * (1 - selected_alphas * selected_decision)

        this_canvas = selected_canvas_patch.permute(0, 3, 1, 4, 2, 5).contiguous()
        # this_canvas: b, 3, selected_h, py, selected_w, px
        this_canvas = this_canvas.view(b, 3, selected_h * patch_size_y, selected_w * patch_size_x).contiguous()
        # this_canvas: b, 3, selected_h * py, selected_w * px
        return this_canvas

    def save_intermediate(cur_canvas, img_id):
        if frame_dir is None:
            return
        frame = crop(cur_canvas[:, :, patch_size_y // factor : -patch_size_y // factor,
                                      patch_size_x // factor : -patch_size_x // factor], original_h, original_w)
        frame_path = os.path.join(frame_dir, f'_L{layer_id:03d}_S{img_id:03d}.jpg')
        save_img(frame[0], frame_path)

    if (even_idx_y.shape[0] > 0) \
    and (even_idx_x.shape[0] > 0):
        for i in range(s):
            canvas = partial_render(cur_canvas, even_y_even_x_coord_y, even_y_even_x_coord_x, i)
            if not is_odd_y:
                canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
            if not is_odd_x:
                canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
            cur_canvas = canvas

            img_id += 1
            save_intermediate(cur_canvas, img_id)

    if (odd_idx_y.shape[0] > 0) \
    and (odd_idx_x.shape[0] > 0):
        for i in range(s):
            canvas = partial_render(cur_canvas, odd_y_odd_x_coord_y, odd_y_odd_x_coord_x, i)
            canvas = torch.cat([cur_canvas[:, :, :patch_size_y // 2, -canvas.shape[3]:], canvas], dim=2)
            canvas = torch.cat([cur_canvas[:, :, -canvas.shape[2]:, :patch_size_x // 2], canvas], dim=3)
            if is_odd_y:
                canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
            if is_odd_x:
                canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
            cur_canvas = canvas

            img_id += 1
            save_intermediate(cur_canvas, img_id)

    if (even_idx_x.shape[0] > 0) \
    and (odd_idx_y.shape[0] > 0):
        for i in range(s):
            canvas = partial_render(cur_canvas, odd_y_even_x_coord_y, odd_y_even_x_coord_x, i)
            canvas = torch.cat([cur_canvas[:, :, :patch_size_y // 2, :canvas.shape[3]], canvas], dim=2)
            if is_odd_y:
                canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
            if not is_odd_x:
                canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
            cur_canvas = canvas

            img_id += 1
            save_intermediate(cur_canvas, img_id)

    if (even_idx_y.shape[0] > 0) \
    and (odd_idx_x.shape[0] > 0):
        for i in range(s):
            canvas = partial_render(cur_canvas, even_y_odd_x_coord_y, even_y_odd_x_coord_x, i)
            canvas = torch.cat([cur_canvas[:, :, :canvas.shape[2], :patch_size_x // 2], canvas], dim=3)
            if not is_odd_y:
                canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, -canvas.shape[3]:]], dim=2)
            if is_odd_x:
                canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
            cur_canvas = canvas

            img_id += 1
            save_intermediate(cur_canvas, img_id)

    cur_canvas = cur_canvas[:, :, patch_size_y // 4:-patch_size_y // 4, 
                                  patch_size_x // 4:-patch_size_x // 4]
    return cur_canvas


def param2img_parallel(param, decision, brushes, cur_canvas):
    """
    Input stroke parameters and decisions for each patch, brushes, current canvas, frame directory,
    and whether there is a border (if intermediate painting results are required).
    
    Output the painting results of adding the corresponding strokes on the current canvas.

    Args:
        param: a tensor with shape 
                batch size x patch along height dimension x patch along width dimension x n_stroke_per_patch x n_param_per_stroke
        decision: a binary tensor with shape 
                batch size x patch along height dimension x patch along width dimension x n_stroke_per_patch
        brushes: a tensor with shape 2 x 3 x brush_height x brush_width.
                On the batch dimension, the 1st slice denotes vertical brush and 
                                        the 2nd one denotes horizontal brush.
        cur_canvas: a tensor with shape batch size x 3 x H x W,
                where H and W denote height and width of padded results of original images.

    Returns:
        cur_canvas: a tensor with shape batch size x 3 x H x W, denoting painting results.
    """
    # param   : b, h, w, num_strokes_per_patch, num_params_per_stroke
    # decision: b, h, w, num_strokes_per_patch
    b, h, w, s, p = param.shape

    param    =    param.view(-1, 8).contiguous()
    decision = decision.view(-1).contiguous().bool()
    
    H, W = cur_canvas.shape[-2:]
    device = cur_canvas.device

    is_odd_y = h % 2 == 1
    is_odd_x = w % 2 == 1
    
    patch_size_y = 2 * H // h
    patch_size_x = 2 * W // w

    even_idx_y = torch.arange(0, h, 2, device=cur_canvas.device)
    even_idx_x = torch.arange(0, w, 2, device=cur_canvas.device)
    odd_idx_y = torch.arange(1, h, 2, device=cur_canvas.device)
    odd_idx_x = torch.arange(1, w, 2, device=cur_canvas.device)

    even_y_even_x_coord_y, even_y_even_x_coord_x = torch.meshgrid([even_idx_y, even_idx_x])
    even_y_odd_x_coord_y, even_y_odd_x_coord_x = torch.meshgrid([even_idx_y, odd_idx_x])
    odd_y_even_x_coord_y, odd_y_even_x_coord_x = torch.meshgrid([odd_idx_y, even_idx_x])
    odd_y_odd_x_coord_y, odd_y_odd_x_coord_x = torch.meshgrid([odd_idx_y, odd_idx_x])

    cur_canvas = F.pad(cur_canvas, [patch_size_x // 4, patch_size_x // 4,
                                    patch_size_y // 4, patch_size_y // 4, 0, 0, 0, 0])

    foregrounds = torch.zeros(param.shape[0], 3, patch_size_y, patch_size_x, device=device)
    alphas      = torch.zeros(param.shape[0], 3, patch_size_y, patch_size_x, device=device)
    
    valid_foregrounds, \
    valid_alphas = param2stroke(param[decision, :], patch_size_y, patch_size_x, brushes)
    
    foregrounds[decision, :, :, :] = valid_foregrounds
    alphas[decision, :, :, :] = valid_alphas

    # foreground, alpha: b * h * w * stroke_per_patch, 3, patch_size_y, patch_size_x
    foregrounds = foregrounds.view(-1, h, w, s, 3, patch_size_y, patch_size_x).contiguous()
    alphas      =      alphas.view(-1, h, w, s, 3, patch_size_y, patch_size_x).contiguous()

    # foreground, alpha: b, h, w, stroke_per_patch, 3, render_size_y, render_size_x
    decision = decision.view(-1, h, w, s, 1, 1, 1).contiguous()

    # decision: b, h, w, stroke_per_patch, 1, 1, 1

    def partial_render(this_canvas, patch_coord_y, patch_coord_x):

        canvas_patch = F.unfold(this_canvas, (patch_size_y, patch_size_x),
                                    stride=(patch_size_y // 2, patch_size_x // 2))

        # canvas_patch: b, 3 * py * px, h * w
        canvas_patch = canvas_patch.view(b, 3, patch_size_y, patch_size_x, h, w).contiguous()
        canvas_patch = canvas_patch.permute(0, 4, 5, 1, 2, 3).contiguous()

        # canvas_patch: b, h, w, 3, py, px
        selected_canvas_patch = canvas_patch[:, patch_coord_y, patch_coord_x, :, :, :]
        selected_foregrounds  =  foregrounds[:, patch_coord_y, patch_coord_x, :, :, :, :]
        selected_alphas       =       alphas[:, patch_coord_y, patch_coord_x, :, :, :, :]
        selected_decisions    =     decision[:, patch_coord_y, patch_coord_x, :, :, :, :]

        for i in range(s):
            cur_foreground = selected_foregrounds[:, :, :, i, :, :, :]
            cur_alpha      =      selected_alphas[:, :, :, i, :, :, :]
            cur_decision   =   selected_decisions[:, :, :, i, :, :, :]
            selected_canvas_patch = cur_foreground * cur_alpha * cur_decision + \
                                    selected_canvas_patch * (1 - cur_alpha * cur_decision)

        this_canvas = selected_canvas_patch.permute(0, 3, 1, 4, 2, 5).contiguous()
        # this_canvas: b, 3, h_half, py, w_half, px
        h_half = this_canvas.shape[2]
        w_half = this_canvas.shape[4]
        this_canvas = this_canvas.view(b, 3, h_half * patch_size_y, 
                                             w_half * patch_size_x).contiguous()
        # this_canvas: b, 3, h_half * py, w_half * px
        return this_canvas

    if even_idx_y.shape[0] > 0 and even_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, even_y_even_x_coord_y, even_y_even_x_coord_x)
        if not is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if not is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if odd_idx_y.shape[0] > 0 and odd_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, odd_y_odd_x_coord_y, odd_y_odd_x_coord_x)
        canvas = torch.cat([cur_canvas[:, :, :patch_size_y // 2, -canvas.shape[3]:], canvas], dim=2)
        canvas = torch.cat([cur_canvas[:, :, -canvas.shape[2]:, :patch_size_x // 2], canvas], dim=3)
        if is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if odd_idx_y.shape[0] > 0 and even_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, odd_y_even_x_coord_y, odd_y_even_x_coord_x)
        canvas = torch.cat([cur_canvas[:, :, :patch_size_y // 2, :canvas.shape[3]], canvas], dim=2)
        if is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, :canvas.shape[3]]], dim=2)
        if not is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    if even_idx_y.shape[0] > 0 and odd_idx_x.shape[0] > 0:
        canvas = partial_render(cur_canvas, even_y_odd_x_coord_y, even_y_odd_x_coord_x)
        canvas = torch.cat([cur_canvas[:, :, :canvas.shape[2], :patch_size_x // 2], canvas], dim=3)
        if not is_odd_y:
            canvas = torch.cat([canvas, cur_canvas[:, :, -patch_size_y // 2:, -canvas.shape[3]:]], dim=2)
        if is_odd_x:
            canvas = torch.cat([canvas, cur_canvas[:, :, :canvas.shape[2], -patch_size_x // 2:]], dim=3)
        cur_canvas = canvas

    cur_canvas = cur_canvas[:, :, patch_size_y // 4:-patch_size_y // 4, 
                                  patch_size_x // 4:-patch_size_x // 4]
    return cur_canvas


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

        final_result = param2img_sequential(param, decision, brushes, final_result, 
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

    final_result = param2img_sequential(param, decision, brushes, final_result, 
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
    brushes = load_brushes()

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
    images_list = [Image.open(f) for f in glob(f'{out_dir}/*.jpg')]
    make_gif(images_list, out_path=f'{out_dir}/out.gif')


