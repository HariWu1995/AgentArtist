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
ACTION_BUNDLE = 5


#################################
#       Functionalities         #
#################################

def load_models(path_to_painter: str, 
                path_to_renderer: str = None):

    # Load painter (actor) model -> ResNet 18
    #               action bundle = 5, 
    #     quadratic Bézier curve = 10,
    #               color channel = 3,
    #                  => outputs = 5 * (10 + 3) = 65
    painter = ResNet(num_inputs=9, depth=18, num_outputs=65)
    painter.load_state_dict(torch.load(path_to_painter), strict=False)
    painter = painter.to(device=DEVICE).eval()

    # Load neural renderer
    if path_to_renderer is not None:
        renderer = FCN()
        renderer.load_state_dict(torch.load(path_to_renderer), strict=False)
        renderer = renderer.to(device=DEVICE).eval()
    else:
        renderer = None

    return painter, renderer


def render(actions, width, canvas, renderer = None):

    # actions = actions.view(-1, 10 + 3)
    stroke_params = actions[:, :10]

    if renderer is not None:
        # Neural rendering
        stroke_new = renderer(stroke_params)

    else:
        # Algorithmic rendering
        stroke_params = stroke_params.detach().cpu().numpy().tolist()
        stroke_new = []
        for param in stroke_params:
            stroke_draw = draw_curve(param, width)
            stroke_draw = torch.tensor(stroke_draw)
            stroke_new.append(stroke_draw)
        stroke_new = torch.stack(stroke_new, dim=0).to(device=DEVICE)

    stroke = 1 - stroke_new
    stroke = stroke.view(-1, width, width, 1)

    colorgb = stroke * actions[:, -3:].view(-1, 1, 1, 3)

    stroke  =  stroke.permute(0, 3, 1, 2).view(-1, ACTION_BUNDLE, 1, width, width)
    colorgb = colorgb.permute(0, 3, 1, 2).view(-1, ACTION_BUNDLE, 3, width, width)

    re5ult = []
    for i in range(ACTION_BUNDLE):
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
        width: int = WIDTH, 
        division: int = 4,
       lowres_step: int = 50,
        hires_step: int = 10,
        verbose: bool = False,
        out_dir: str = './results',
    ):

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    H, W = image.shape[:2]
    T, C = init_aux(WIDTH)
    
    img_count = 0
    img_stack = []      # to generate animation
    guidelowres = []    # to reproduce w/o running model
    guidehires = []

    # if division > 1:
    #     max_step = max_step // 2

    #########################
    #       Low-res         #
    #########################

    canvas = torch.zeros([1, 3, width, width]).to(device=DEVICE)
    patch_img, \
    sized_img = preprocess(image, width, division, device=DEVICE, hires=False)
    
    print("\n Low-res painting ...")
    lowres_bar = tqdm(range(lowres_step))
    for i in lowres_bar:

        # Inference
        step = T * i / lowres_step  # step encoding -> dense vector
        inputs = torch.cat([canvas, sized_img, step, C], dim=1)
        actions = painter(inputs).view(-1, 10 + 3)
        canvas, re5ult = render(actions, width, canvas, renderer)
        guidelowres.append(actions)

        # Evaluation
        loss = ((canvas - sized_img) ** 2).mean()
        log = f'step {i}, L2Loss = {loss:.5f}'
        lowres_bar.set_description(log)
        
        # Postprocessing
        for res in re5ult:
            res = postprocess(res, out_size=(H, W))
            cv2.imwrite(f'{out_dir}/out_{img_count:03d}.png', res)
            img_stack.append(res)
            img_count += 1

    # Guidance
    guidelowres = torch.cat(guidelowres, dim=0).detach().cpu()
    guidescale = torch.ones(guidelowres.shape[0], 1)
    guidepatch = -torch.ones(guidelowres.shape[0], 1)
    guidelowres = torch.cat([guidescale, guidepatch, guidelowres], dim=1)

    if verbose:
        print('\nGuidelines:')
        print(guidelowres)
        print(guidelowres.shape)

    if division <= 1:
        return img_stack, guidelowres

    #########################
    #        Hi-res         #
    #########################

    canvas = canvas[0].detach().cpu().numpy()
    canvas = np.transpose(canvas, (1, 2, 0))    
    canvas = preprocess(canvas, width, division, hires=True, device=DEVICE)

    num_patches = division * division
    C = C.expand(num_patches, 2, width, width)
    T = T.expand(num_patches, 1, width, width)

    print("\n Hi-res painting ...")
    hires_bar = tqdm(range(hires_step))
    for i in hires_bar:

        # Inference
        step = T * i / hires_step
        inputs = torch.cat([canvas, patch_img, step, C], dim=1)
        actions = painter(inputs).view(-1, 10 + 3)
        canvas, re5ult = render(actions, width, canvas, renderer)
        guidehires.append(actions)

        # Evaluation
        loss = ((canvas - patch_img) ** 2).mean()
        log = f'step {i}, L2Loss = {loss:.5f}'
        hires_bar.set_description(log)

        # Postprocessing
        for res in re5ult:
            res = postprocess(res, width, division, out_size=(H, W))
            cv2.imwrite(f'{out_dir}/out_{img_count:03d}.png', res)
            img_stack.append(res)
            img_count += 1

    # Guidance
    guidehires = torch.cat(guidehires, dim=0).detach().cpu()
    guidescale = torch.ones(guidehires.shape[0], 1) * division
    guidepatch = torch.arange(num_patches).repeat(ACTION_BUNDLE).view(ACTION_BUNDLE, -1)\
                                .T.flatten().repeat(hires_step).view(-1, 1)
    guidehires = torch.cat([guidescale, guidepatch, guidehires], dim=1)

    if verbose:
        print('\nGuidelines:')
        print(guidehires)
        print(guidehires.shape)

    guidelines = torch.cat([guidelowres, guidehires], dim=0).numpy().tolist()

    return img_stack, guidelines


if __name__ == "__main__":

    # Load models
    painter_ckpt_path = './checkpoints/paint_ddpg_sbr/default/actor.pkl'
    renderer_ckpt_path = './checkpoints/paint_ddpg_sbr/default/renderer.pkl'
    painter, renderer = load_models(painter_ckpt_path)

    # Load image
    image_path = "C:/Users/Mr. RIAH/Pictures/_character/Nancy-Closeup.jpg"
    # image_path = "./samples/van-gogh-garden-at-arles.png"
    image_size = 1024
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # image = cv2.copyMakeBorder(image, (852-552)//2, (852-552)//2, 0, 0, cv2.BORDER_CONSTANT)
    image = cv2.resize(image, (image_size, image_size))

    # Run pipeline
    division = int(image_size / WIDTH)
    out_dir = f'./results/nancy_{image_size}'
    with torch.no_grad():
        images_list, \
        guidelines = run_pipeline(painter, renderer, image,
                                    width = WIDTH, division = division,
                                    verbose = True, out_dir = out_dir, 
                                hires_step = 20, lowres_step = 50)

    # Save guidance
    #   num_lines = num_actions * (lowres_step + hires_step * (resolution / width)**2)
    #             =       5     * (     50     +      20    * (   1024    /   128)**2) = 6650
    print('\nSaving guidelines ...')
    with open(f'{out_dir}/guidelines.csv', 'w') as fwriter:
        fwriter.write('step,division,patch,x0,y0,x1,y1,x2,y2,z0,z2,w0,w2,r,g,b')
        for i, param in enumerate(guidelines):
            param = [str(i+1)] + [str(int(p)) for p in param[:2]] \
                               + ['%.5f' % p for p in param[2:]] 
            fwriter.write('\n' + ','.join(param))

    # Animation
    #   num_frames = num_actions * (lowres_step + hires_step)
    #              =    5        * (      50    +     20    ) = 350
    print('\nMaking GIF ...')
    make_gif(images_list[::2] + \
            [images_list[-1]]*5, out_path=f'{out_dir}/out.gif')

