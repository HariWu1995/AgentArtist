import os
import itertools

from glob import glob
from tqdm import tqdm

import cv2
import numpy as np


if __name__ == "__main__":

    image_size = 512

    root_dir = 'F:/Document/Artwork/_ghostories_/output'
    out_dir = f'{root_dir}/duong-di-ha-giang-1-picasso_{image_size}L_doublew_3x4/woCx8'

    # (255, 255, 255) for white, (0, 0, 0) for black
    bg_color = (255, 255, 255)

    # 108 (W) x 146 (H) ~ 1:1.35185
    # 1024 * 1.35185 = 1384 < 1536
    width = 1024    
    height = 1536   
    canvas = np.full((height, width, 3), bg_color, dtype=np.uint8)

    rough_frames = 1380
    fined_frames = 2760

    # Video Writer
    fps = 30
    size = (width, height)
    video_format = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(f'{out_dir}/out_combined.mp4', video_format, fps, size)

    # Rough Painting
    for r, c in itertools.product([2,3,1], [1,2]):
        for i in tqdm(range(rough_frames)):
            
            part_path = f"{out_dir}/R{r}C{c}/out_{i:04d}.png"
            part_frame = cv2.imread(part_path)
            if not isinstance(part_frame, np.ndarray):
                print(r,c,i)
                continue

            y_start = (r-1) * image_size
            y_end = y_start + image_size
            x_start = (c-1) * image_size
            x_end = x_start + image_size

            canvas[y_start:y_end, x_start:x_end] = part_frame
            if (r == 1) and (i > 1000):
                continue
            if (r in [2,3]) and (i > 1250):
                continue
            if i < 5:
                for _ in range(8):
                    video_writer.write(canvas)
            elif i > 1000:
                if (i+1) % 40 != 0:
                    continue
            elif i > 100:
                if (i+1) % 12 != 0:
                    continue
            video_writer.write(canvas)

    # Refined Painting
    for i in tqdm(range(rough_frames, fined_frames)):
        for r, c in itertools.product([2,3,1], [1,2]):
            part_path = f"{out_dir}/R{r}C{c}/out_{i:04d}.png"
            part_frame = cv2.imread(part_path)
            if not isinstance(part_frame, np.ndarray):
                print(r,c,i)
                continue

            y_start = (r-1) * image_size
            y_end = y_start + image_size
            x_start = (c-1) * image_size
            x_end = x_start + image_size
            canvas[y_start:y_end, x_start:x_end] = part_frame

        if 1380 <= i <= 1385:
            for _ in range(10):
                video_writer.write(canvas)
        elif i > 1500:
            if (i+1) % 5 != 0:
                continue
        elif i > 2000:
            if (i+1) % 20 != 0:
                continue
        video_writer.write(canvas)

    for _ in range(10):
        video_writer.write(canvas)

    # Finalize
    video_writer.release()
    cv2.imwrite(f'{out_dir}/out_combined.png', canvas)

    # Run CMD
    # python -m modeling.paint_doublew.reproduction

