import os

import torch
import torch.nn.functional as F

from modeling.paint_transformer.brushstrokes import drawing
from modeling.paint_transformer.utilities import save_img, pad, crop


def render_sequential(
        param, decision, brushes, cur_canvas, 
        frame_dir: str = None, has_border: bool = False, 
        original_h: int = None, original_w: int = None, 
        layer_id: int = 0, save_mode: str = 'combined',
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

    cur_canvas = F.pad(cur_canvas, [patch_size_x // 4] * 2 + \
                                   [patch_size_y // 4] * 2 + [0] * 4)

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
                 selected_alphas[selected_decision, :, :, :] = drawing(selected_param[selected_decision, :], 
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

            img_id += 1
            if save_mode == 'partial':
                save_intermediate(canvas, img_id)
            else:
                cur_canvas = canvas
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

            img_id += 1
            if save_mode == 'partial':
                save_intermediate(canvas, img_id)
            else:
                cur_canvas = canvas
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

            img_id += 1
            if save_mode == 'partial':
                save_intermediate(canvas, img_id)
            else:
                cur_canvas = canvas
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

            img_id += 1
            if save_mode == 'partial':
                save_intermediate(canvas, img_id)
            else:
                cur_canvas = canvas
                save_intermediate(cur_canvas, img_id)

    cur_canvas = cur_canvas[:, :, patch_size_y // 4:-patch_size_y // 4, 
                                  patch_size_x // 4:-patch_size_x // 4]
    return cur_canvas


def render_parallel(param, decision, brushes, cur_canvas):
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

    foregrounds = torch.zeros(param.shape[0], 3, patch_size_y, patch_size_x, device=device)
    alphas      = torch.zeros(param.shape[0], 3, patch_size_y, patch_size_x, device=device)
    
    valid_foregrounds, \
    valid_alphas = drawing(param[decision, :], patch_size_y, patch_size_x, brushes)
    
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


