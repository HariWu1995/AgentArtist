import os
from tqdm import tqdm

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.paint_ddpg_sbr.baseline.Renderer.model import FCN
from modeling.paint_ddpg_sbr.baseline.DRL.actor import ResNet
from modeling.paint_ddpg_sbr.brushstrokes import draw_curve
from modeling.paint_ddpg_sbr.utilities import preprocess, postprocess, make_gif


#################################
#       Global variables        #
#################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

WIDTH = 128
SEQ_LEN = 5


#################################
#       Functionalities         #
#################################

def load_models(path_to_painter: str, 
                path_to_renderer: str, ):

    # Load painter (actor) model -> ResNet 18
    #               action_bundle = 5, 
    #     quadratic Bézier curve = 10,
    #               color channel = 3,
    #                  => outputs = 5 * (10 + 3) = 65
    painter = ResNet(num_inputs=9, depth=18, num_outputs=65)
    painter.load_state_dict(torch.load(path_to_painter), strict=False)

    # Load neural renderer
    renderer = FCN()
    renderer.load_state_dict(torch.load(path_to_renderer), strict=False)

    # Send to device
    painter = painter.to(device=DEVICE).eval()
    renderer = renderer.to(device=DEVICE).eval()

    return painter, renderer


def render(x, width, canvas, renderer):
    
    # b * (10 + 3)
    x = x.view(-1, 10 + 3)

    stroke = 1 - renderer(x[:, :10])
    stroke = stroke.view(-1, width, width, 1)

    colorgb = stroke * x[:, -3:].view(-1, 1, 1, 3)

    stroke = stroke.permute(0, 3, 1, 2)
    colorgb = colorgb.permute(0, 3, 1, 2)

    stroke = stroke.view(-1, SEQ_LEN, 1, width, width)
    colorgb = colorgb.view(-1, SEQ_LEN, 3, width, width)

    re5ult = []
    for i in range(SEQ_LEN):
        canvas = canvas * (1 - stroke[:, i]) + colorgb[:, i]
        re5ult.append(canvas)
    return canvas, re5ult


def init_aux(width: int = WIDTH):
    # Step number
    T = torch.ones([1, 1, width, width], dtype=DTYPE).to(device=DEVICE)
    
    # CoordConv
    C = torch.zeros([1, 2, width, width])
    for i in range(width):
        for j in range(width):
            C[0, 0, i, j] = i / (width - 1.)
            C[0, 1, i, j] = j / (width - 1.)
    C = C.to(device=DEVICE)

    return T, C


def run_pipeline(
        painter, 
        renderer,
        image: np.ndarray,
        width: int = 128, 
        division: int = 4,
        lowres_step: int = 50,
        hires_step: int = 10,
        verbose: bool = False,
        out_dir: str = './results',
    ):

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    H, W = image.shape[:2]
    T, C = init_aux(width)
    
    num_canvas = division * division
    img_count = 0
    img_stack = []

    # if division > 1:
    #     max_step = max_step // 2

    # Low-res
    canvas = torch.zeros([1, 3, width, width]).to(device=DEVICE)
    patch_img, \
    sized_img = preprocess(image, num_canvas, width, division, hires=False, device=DEVICE)
    
    print("\n Low-res painting ...")
    lowres_bar = tqdm(range(lowres_step))
    for i in lowres_bar:
        step = T * i / lowres_step
        inputs = torch.cat([canvas, sized_img, step, C], dim=1)
        actions = painter(inputs)
        canvas, re5ult = render(actions, width, canvas, renderer)
        
        loss = ((canvas - sized_img) ** 2).mean()
        log = f'step {i}, L2Loss = {loss:.5f}'
        lowres_bar.set_description(log)
        
        for res in re5ult:
            res = postprocess(res, out_size=(H, W), is_divided=False)
            cv2.imwrite(f'{out_dir}/out_{img_count:03d}.png', res)
            img_stack.append(res)
            img_count += 1

    if division <= 1:
        return img_stack

    # Hires
    canvas = canvas[0].detach().cpu().numpy()
    canvas = np.transpose(canvas, (1, 2, 0))    
    canvas = preprocess(canvas, num_canvas, width, division, hires=True, device=DEVICE)

    C = C.expand(num_canvas, 2, width, width)
    T = T.expand(num_canvas, 1, width, width)

    print("\n Hi-res painting ...")
    hires_bar = tqdm(range(hires_step))
    for i in hires_bar:
        step = T * i / hires_step
        inputs = torch.cat([canvas, patch_img, step, C], dim=1)
        actions = painter(inputs)
        canvas, re5ult = render(actions, width, canvas, renderer)

        loss = ((canvas - patch_img) ** 2).mean()
        log = f'step {i}, L2Loss = {loss:.5f}'
        hires_bar.set_description(log)

        for res in re5ult:
            res = postprocess(res, width, division, out_size=(H, W), is_divided=True)
            cv2.imwrite(f'{out_dir}/out_{img_count:03d}.png', res)
            img_stack.append(res)
            img_count += 1

    return img_stack


if __name__ == "__main__":

    # Load models
    painter_ckpt_path = './checkpoints/paint_ddpg_sbr/default/actor.pkl'
    renderer_ckpt_path = './checkpoints/paint_ddpg_sbr/default/renderer.pkl'
    painter, renderer = load_models(painter_ckpt_path, renderer_ckpt_path)

    # Load image
    # image_path = "C:/Users/Mr. RIAH/Pictures/_character/Nancy-Closeup.jpg"
    image_path = "./samples/van-gogh-garden-at-arles.png"
    image_size = 1024
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.copyMakeBorder(image, (852-552)//2, (852-552)//2, 0, 0, cv2.BORDER_CONSTANT)
    image = cv2.resize(image, (image_size, image_size))

    # Run pipeline
    D = int(image_size / WIDTH)
    out_dir = f'./results/van_gogh_02_{image_size}'
    with torch.no_grad():
        images_list = run_pipeline(painter, renderer, image,
                                    width = WIDTH, division = D,
                                  verbose = True, out_dir = out_dir, 
                                hires_step = 20, lowres_step = 50)

    # Animation
    make_gif(images_list, out_path=f'{out_dir}/out.gif')

