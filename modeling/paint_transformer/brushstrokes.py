import torch
import torch.nn.functional as F

from modeling.paint_transformer.utilities import read_img
from modeling.paint_transformer.morphology import erosion, dilation


BRUSH_FOLDER = './checkpoints/paint_transformer/brush'
BRUSH_TYPE = 'brush_large'


def load_brushes(brush_type: str = BRUSH_TYPE):
    brush_v = read_img(f'{BRUSH_FOLDER}/{brush_type}_vertical.png', 'L')
    brush_h = read_img(f'{BRUSH_FOLDER}/{brush_type}_horizontal.png', 'L')
    brushes = torch.cat([brush_v, brush_h], dim=0)
    return brushes


def drawing(param, H, W, brushes):
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
    b = param.shape[0]
    device = param.device

    # Firstly, resize the brushes to the required shape,
    # in order to decrease GPU memory especially when the required shape is small.
    brushes = F.interpolate(brushes, (H, W))

    # Extract shape parameters and color parameters.
    param_list = torch.split(param, 1, dim=1)
    x0, y0, w, h, theta = [p.squeeze(-1) for p in param_list[:5]]
    R, G, B = param_list[5:]

    # Pre-compute sin theta and cos theta
    sin_theta = torch.sin(torch.acos(torch.tensor(-1., device=device)) * theta)
    cos_theta = torch.cos(torch.acos(torch.tensor(-1., device=device)) * theta)

    # index indicates each stroke should be in which direction (vertical or horizontal).
    # When h > w, vertical stroke should be used. 
    # When h <= w, horizontal stroke should be used.
    index = torch.full((b,), -1, device=device, dtype=torch.long)
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

