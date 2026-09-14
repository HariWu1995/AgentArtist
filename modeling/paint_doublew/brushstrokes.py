import torch
import torch.nn.functional as F

from modeling.paint_doublew.utilities import read_img
from modeling.paint_doublew.compositor.morphology import erosion, dilation


BRUSH_FOLDER = './checkpoints/paint_doublew/brush'
BRUSH_TYPE = 'brush_large'


def load_brushes(brush_type: str = BRUSH_TYPE, pad: bool = False):
    if not pad:
        brush_v = read_img(f'{BRUSH_FOLDER}/{brush_type}_vertical.png', 'L')
        brush_h = read_img(f'{BRUSH_FOLDER}/{brush_type}_horizontal.png', 'L')
    else:
        brush_v = read_img(f'{BRUSH_FOLDER}/{brush_type}_vertical_pad.png', 'L')
        brush_h = read_img(f'{BRUSH_FOLDER}/{brush_type}_horizontal_pad.png', 'L')
    brushes = torch.cat([brush_v, brush_h], dim=0)
    return brushes

