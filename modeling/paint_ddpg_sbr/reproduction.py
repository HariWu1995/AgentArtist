from typing import List, Tuple, Union
from tqdm import tqdm

import pandas as pd
import numpy as np
import cv2

from modeling.paint_ddpg_sbr.brushstrokes import draw_curve
from modeling.paint_ddpg_sbr.utilities import make_video


WIDTH = 128


def reproduce(
        guidelines: pd.DataFrame, 
        video_path: str = None,
        frame_size: int = 512,
        frame_per_sec: int = 5, 
    ):
    """
    Arguments:
        guidelines: List of params (step,division,patch,x0,y0,x1,y1,x2,y2,z0,z2,w0,w2,r,g,b)
                        where   (step, division, patch) are metadata
                                (x0,y0,x1,y1,x2,y2,z0,z2,w0,w2) are input of drawing function
                                (r,g,b) are color channels
    """
    scale = guidelines.division.max()
    full_canvas = np.zeros([WIDTH * scale, WIDTH * scale, 3]).astype('float32')
    
    video_fps = frame_per_sec
    video_size = (frame_size, frame_size)
    video_writer = None

    if isinstance(video_path, str):
        # print("\nInitializing video writer ...")
        video_format = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, video_format, video_fps, video_size)
        init_frame = ((full_canvas + 1) * 255).astype(np.uint8)
        init_frame = cv2.resize(init_frame, video_size, interpolation=cv2.INTER_AREA)
        video_writer.write(init_frame)

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

        if video_writer is not None:
            video_frame = (full_canvas * 255).astype(int)
            video_frame = np.clip(video_frame, 0, 255).astype(np.uint8)
            video_frame = cv2.resize(video_frame, video_size, interpolation=cv2.INTER_AREA)
            video_writer.write(video_frame)

    if video_writer is not None:
        video_writer.release()

    full_canvas = (full_canvas * 255).astype(int)
    return full_canvas


if __name__ == "__main__":

    gen_dir = f'F:/Document/Artwork/_ghostories_/output/duong-di-ha-giang-1-flatillustration_2048'
    video_path = f'{gen_dir}/out.mp4'

    video_fps = 20
    video_size = (2048, 2048)
    video_format = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, video_format, video_fps, video_size)

    for i in tqdm(range(100, 1000)):
        image_path = f"{gen_dir}/out_{i:03d}.png"
        image = cv2.imread(image_path)
        frame = cv2.resize(image, video_size, interpolation=cv2.INTER_AREA)
        video_writer.write(frame)

    video_writer.release()
    quit()

    guidelines = pd.read_csv(f'{gen_dir}/guidelines.csv').set_index('step')
    print(guidelines)

    canvas = reproduce(guidelines, video_path)
    cv2.imwrite(f'{gen_dir}/out_reproduced.png', canvas)

    # Animation
    # print('\nMaking Video ...')
    # make_video(all_frames, path2video)

    # Run CMD
    # python -m modeling.paint_ddpg_sbr.reproduction

