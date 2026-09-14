import os
from tqdm import tqdm
from PIL import Image

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from modeling.paint_doublew.brushstrokes import load_brushes
from modeling.paint_doublew.compositor.DRL.actor import ResNet
from modeling.paint_doublew.compositor.Renderer import FCN

from .utilities import (
    PARAM_NUM, WIDTH, WIDTH_OUT,
    transform_img, resize_64, resize_128, resize_256, resize_512,
    small2large, large2small, 
    decode3, decode, final_decode,
)


#################################
#       Global variables        #
#################################

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


#################################
#       Functionalities         #
#################################

def load_models(
    path_to_painter: str = None, 
    path_to_compositor: str = None,
    path_to_renderer: str = None,
):
    # Load painter (actor) model -> ResNet 18
    #               action bundle = 5, 
    #               color channel = 3,
    #                  => outputs = 5 * (5 + 3) = 65
    if path_to_painter is not None:
        painter = ResNet(num_inputs=6, depth=18, num_outputs=5*(PARAM_NUM+3))
        painter.load_state_dict(torch.load(path_to_painter), strict=False)
        painter = painter.to(device=DEVICE).eval()
    else:
        painter = None

    if path_to_compositor is not None:
        compositor = ResNet(num_inputs=6, depth=18, num_outputs=4) # canvas, target
        compositor.load_state_dict(torch.load(path_to_compositor), strict=False)
        compositor = compositor.to(device=DEVICE).eval()
    else:
        compositor = None

    # Load neural renderer
    if path_to_renderer is not None:
        renderer = FCN(PARAM_NUM, need_alphas=True, need_edge=True)
        renderer.load_state_dict(torch.load(path_to_renderer), strict=False)
        renderer = renderer.to(device=DEVICE).eval()
    else:
        renderer = None

    return painter, compositor, renderer


def run_pipeline_by_size(
    image,
    num_strokes: int,
    Painter,
    Compositor, 
    meta_brushes,
    background_color: str = 'black',
):
    # Size 512 x 512
    WIDTH = 512
    steps = num_strokes // 5
    recursive_number = 1

    canvas = torch.zeros([1, 3, WIDTH, WIDTH]).to(device=DEVICE)
    if background_color.lower() == 'white':
        canvas += 1.

    if steps > 200:
        recursive_number = steps // 200 + 1
        steps = steps // recursive_number + 1
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    # image = Image.open(image_path).convert('RGB')
    image = transform_img(image).unsqueeze(0).to(device=DEVICE)
    image = resize_512(image)
    image = image[:, [2,1,0]]
    
    boxes0 = []
    boxes1 = []
    params = []

    losses = 0
    loss_fn = torch.nn.MSELoss()
    
    for i in tqdm(range(steps)):
        box = Compositor(torch.cat([canvas, image], dim=1))
        boxes0.append(box[0].detach())

        x1, y1, x2, y2 = torch.round(box[0] * 511).detach().int()
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        resize = transforms.Resize(((x2 + 1 - x1), (y2 + 1 - y1)))
        tar_canvas_box = resize_512( image[0, :, x1:x2 + 1, y1:y2 + 1]).unsqueeze(0)
        tmp_canvas_box = resize_512(canvas[0, :, x1:x2 + 1, y1:y2 + 1]).unsqueeze(0)
        
        for j in range(recursive_number):
            actions = Compositor(torch.cat([tmp_canvas_box, tar_canvas_box], 1))
            boxes1.append(actions[0].detach())
            tmp_canvas_box, params = decode(Painter, meta_brushes, actions, params,
                                            tmp_canvas_box, tar_canvas_box)

        canvas[0, :, x1:x2+1, y1:y2+1] = resize(tmp_canvas_box[0])

    pixel_loss = loss_fn(canvas, image)
    losses += float(pixel_loss.detach())
    print('MSE Distance: ', losses)

    canvas = torch.zeros([1, 3, WIDTH, WIDTH]).to(device=DEVICE)
    if background_color.lower() == 'white':
        canvas += 1.
    canvas, images_list = final_decode(canvas, boxes0, boxes1, params, 
                                        meta_brushes, recursive_number)
    return canvas, images_list, boxes0, boxes1, params


def run_pipeline_by_block(
    image,
    num_strokes: int,
    Painter, 
    meta_brushes,
    background_color: str = 'black',
):
    # Block: 5 x 5
    K = 5
    canvas_cnt = K * K
    origin_shape = (512, 512)

    WIDTH = 128

    canvas = torch.zeros([1, 3, WIDTH, WIDTH]).to(device=DEVICE)
    if background_color.lower() == 'white':
        canvas += 1.

    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    patch_img = cv2.resize(image, (WIDTH * K, WIDTH * K))
    patch_img =  large2small(patch_img, WIDTH, K, canvas_cnt)
    patch_img = np.transpose(patch_img, (0, 3, 1, 2))
    patch_img = torch.tensor(patch_img).to(device=DEVICE).float() / 255.

    image = cv2.resize(image, (WIDTH, WIDTH))
    image = image.reshape(1, WIDTH, WIDTH, 3)
    image = np.transpose(image, (0, 3, 1, 2))
    image = torch.tensor(image).to(device=DEVICE).float() / 255.

    steps = num_strokes // (canvas_cnt + 1)

    for i in range(steps):
        actions = Painter(torch.cat([canvas, image], dim=1))
        canvas, images_list = decode3(actions, canvas, meta_brushes)

    canvas = canvas[0].detach().cpu().numpy()
    canvas = np.transpose(canvas, (1, 2, 0))
    canvas = cv2.resize(canvas, (WIDTH * K, WIDTH * K))
    canvas = large2small(canvas, WIDTH, K, canvas_cnt)
    canvas = np.transpose(canvas, (0, 3, 1, 2))
    canvas = torch.tensor(canvas).to(device=DEVICE).float()
    
    for i in range(steps):
        actions = Painter(torch.cat([canvas, patch_img], dim=1))
        canvas, images_list = decode3(actions, canvas, meta_brushes)
    
    pixel_loss = loss_mse(canvas, patch_img)
    print('MSE Distance: ', pixel_loss)
    
    output = res[-1].detach().cpu().numpy()  # d * d, 3, width, width
    output = np.transpose(output, (0, 2, 3, 1))
    output = small2large(output, WIDTH, K)
    output = (output * 255).astype('uint8')
    output = cv2.resize(output, origin_shape)

    return output, []


def run_pipeline(
    image,
    meta_brushes,
    Painter,
    Compositor,
    num_strokes: int = 5_000,
    background_color: str = 'black',
    out_dir: str = './results',
    video_clip: bool = True,
):
    #    Painter for `run_pipeline_by_size` 
    # Compositor for `run_pipeline_by_block`
    assert (Painter is not None) or (Compositor is not None)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(f'{out_dir}/target.png', image)

    if Compositor is not None:
        canvas, images_list, \
        boxes0, boxes1, params = run_pipeline_by_size(image, num_strokes, Painter, Compositor, meta_brushes, background_color)

    elif Painter is not None:
        canvas, images_list = run_pipeline_by_block(image, num_strokes, Painter, meta_brushes, background_color)

    if video_clip:
        fps = 10
        size = (512, 512)
        video_format = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(f'{out_dir}/out.mp4', video_format, fps, size)

    for i in tqdm(range(len(images_list))):
        if i > 1_000 and i % 25 != 0:
            continue
        elif i > 100 and i % 10 != 0:
            continue
        frame = images_list[i]
        cv2.imwrite(f'{out_dir}/out_{i:04d}.png', frame)
        if video_clip:
            video_writer.write(frame)

    save_image(canvas[:, [2, 1, 0]], 'output.png', nrow=1, normalize=False)

    if video_clip:
        video_writer.release()

    return images_list, (boxes0, boxes1, params)


if __name__ == "__main__":

    # Load models
    painter_ckpt_path = './checkpoints/paint_doublew/painter.pkl'
    renderer_ckpt_path = './checkpoints/paint_doublew/renderer.pkl'
    compositor_ckpt_path = './checkpoints/paint_doublew/compositor.pkl'
    Painter, Compositor, Renderer = load_models(painter_ckpt_path, compositor_ckpt_path, None)

    meta_brushes = load_brushes('brush_small').to(device=DEVICE)

    # Load image
    # image_path = "C:/Users/Mr. RIAH/Pictures/_character/Nancy-Closeup.jpg"
    # image_path = "./samples/van-gogh-garden-at-arles.png"
    image_path = "F:/Document/Artwork/_ghostories_/styles/duong-di-ha-giang-1-picasso.jpg"
    image_size = 512
    image_bg = (255, 255, 255) # (255, 255, 255) for white, (0, 0, 0) for black
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.copyMakeBorder(image, (1536-1365)//2, (1536-1365)//2, 0, 0, borderType=cv2.BORDER_CONSTANT, value=image_bg)
    # image = cv2.resize(image, (image_size, image_size))

    H, W = image.shape[:2]  # H = 1536 = 3 * 512 / W = 2048 = 4 * 512
    r, c = 3, 4
    image = image[512*(r-1):512*r, 512*(c-1):512*c]
    # image = cv2.copyMakeBorder(image, 1024-682, 0, 0, 0, borderType=cv2.BORDER_CONSTANT, value=image_bg)
    image = cv2.resize(image, (image_size, image_size))

    # Run pipeline
    # out_dir = f'./results/nancy_{image_size}'
    out_dir = f'F:/Document/Artwork/_ghostories_/output/duong-di-ha-giang-1-picasso_{image_size}S_doublew_R{r}C{c}'
    
    with torch.no_grad():
        images_list, \
        guidelines = run_pipeline(image, meta_brushes, Painter, Compositor,
                                  out_dir = out_dir, num_strokes = 25_000,
                                                background_color = 'white')

    # Save guidance
    #   num_lines = num_actions * (lowres_step + hires_step * (resolution / width)**2)
    #             =       5     * (     50     +      20    * (   1024    /   128)**2) = 6650
    # print('\nSaving guidelines ...')
    # with open(f'{out_dir}/guidelines.csv', 'w') as fwriter:
    #     fwriter.write('step,division,patch,x0,y0,x1,y1,x2,y2,z0,z2,w0,w2,r,g,b')
    #     for i, param in enumerate(guidelines):
    #         param = [str(i+1)] + [str(int(p)) for p in param[:2]] \
    #                            + ['%.5f' % p for p in param[2:]] 
    #         fwriter.write('\n' + ','.join(param))

    # Animation
    #   num_frames = num_actions * (lowres_step + hires_step)
    #              =    5        * (      50    +     20    ) = 350
    # print('\nMaking GIF ...')
    # frames_list = images_list[:20] + \
    #               images_list[20::10] + \
    #              [images_list[-1]]*5
    # make_gif(frames_list, out_path=f'{out_dir}/out.gif')

    # Run CMD
    # python -m modeling.paint_doublew.inference

