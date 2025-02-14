from typing import List, Tuple, Union
from tqdm import tqdm

import pandas as pd
import numpy as np
import cv2

from modeling.paint_ddpg_sbr.brushstrokes import draw_curve


WIDTH = 128


def reproduce(guidelines: pd.DataFrame):
    """
    Arguments:
        guidelines: List of params (step,division,patch,x0,y0,x1,y1,x2,y2,z0,z2,w0,w2,r,g,b)
                        where   (step, division, patch) are metadata
                                (x0,y0,x1,y1,x2,y2,z0,z2,w0,w2) are input of drawing function
                                (r,g,b) are color channels
    """
    scale = guidelines.division.max()
    full_canvas = np.zeros([WIDTH * scale, WIDTH * scale, 3]).astype('float32')
    
    for step, params in tqdm(guidelines.iterrows(), total=len(guidelines)):
        params = params.values.tolist()

        division, patch = [int(p) for p in params[:2]]
        width = int(WIDTH * (scale / division))

        stroke_params = params[2:-3]
        colorgb = np.array(params[-3:]).reshape(1, 1, 3)

        stroke = draw_curve(stroke_params, width)
        stroke = stroke.reshape(width, width, 1)

        colorgb = (1 - stroke) * colorgb

        if patch == -1:
            full_canvas = full_canvas * stroke + colorgb
        else:
            start_y = (patch // 8) * WIDTH
            start_x = (patch % 8) * WIDTH
            patch_canvas = full_canvas[start_y : start_y + WIDTH, 
                                       start_x : start_x + WIDTH, :]
            patch_canvas = patch_canvas * stroke + colorgb
            full_canvas[start_y : start_y + WIDTH, 
                        start_x : start_x + WIDTH, :] = patch_canvas

    full_canvas = (full_canvas * 255).astype(int)
    return full_canvas


if __name__ == "__main__":

    gen_dir = f'./results/nancy_1024'

    guidelines = pd.read_csv(f'{gen_dir}/guidelines.csv').set_index('step')
    print(guidelines)

    canvas = reproduce(guidelines)
    cv2.imwrite(f'{gen_dir}_reproduced.png', canvas)

