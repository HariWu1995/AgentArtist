import os
from tqdm import tqdm

import math
import torch
import numpy as np
import pandas as pd

from modeling.paint_transformer.rendering import render_sequential as rendering
from modeling.paint_transformer.utilities import read_img, save_img, pad, crop
from modeling.paint_transformer.brushstrokes import load_brushes


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

PATCH_SIZE = 32
STROKE_NUM = 8      # number of strokes per patch per layer
STROKE_PARAM = 5


def reproduce(guidelines: pd.DataFrame):
    """
    Arguments:
        guidelines: List of params (step,decision,layer,size,patch,x0,y0,w,h,theta,r,g,b)
                        where   (step, decision, layer, size, patch) are metadata
                                (x0, y0, w, h, theta) are shape parameters
                                          and (r,g,b) are color channels, 
                                          which are inputs for drawing function
    """
    brushes = load_brushes('brush_large').float().to(device=DEVICE)
    max_size = int(guidelines['size'].max())

    out_dir = './results/reproduction'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    new_canvas = torch.zeros([1, 3, max_size, max_size]).float().to(device=DEVICE)

    layers = sorted([l for l in guidelines['layer'].unique().tolist() if l != -1])
    for layer in tqdm(layers):
        layer_params = guidelines[guidelines['layer'] == layer]

        patch_num = 2 ** layer

        stroke_decision = layer_params['decision'].values
        stroke_params = layer_params[['x0','y0','w','h','theta','r','g','b']].values

        decision = torch.from_numpy(stroke_decision).to(device=DEVICE)
        stparams = torch.from_numpy(stroke_params).to(device=DEVICE)
        
        decision = decision.view(1, patch_num, patch_num, STROKE_NUM   ).contiguous().bool()
        stparams = stparams.view(1, patch_num, patch_num, STROKE_NUM, 8).contiguous().float()

        # Rendering
        _ = rendering(stparams, decision, brushes, new_canvas, 
                        out_dir, False, max_size, max_size, layer, save_mode='partial')

    return new_canvas


if __name__ == "__main__":

    gen_dir = f'./results/van_gogh_1024'

    guidelines = pd.read_csv(f'{gen_dir}/guidelines.csv').set_index('step')
    print(guidelines)

    canvas = reproduce(guidelines)
    # save_img(canvas[0], f'{gen_dir}_reproduced.png')

