"""Image utilities - Mixed pure and impure functions.

This module contains:
- Pure conversion functions imported from deforum.utils.image_utils
- Impure I/O functions for saving/loading frames (kept here due to side effects)
"""

import os

import cv2
from cv2.typing import MatLike

from . import filename_utils
from ..data.render_data import RenderData

# Import pure conversion functions from refactored utils module
from deforum.utils.image_utils import (
    bgr_to_rgb,
    numpy_to_pil,
    pil_to_numpy,
    is_PIL,
)


def save_cadence_frame(data: RenderData, i: int, image: MatLike, is_overwrite: bool = True):
    filename = filename_utils.frame_filename(data, i)
    save_path: str = os.path.join(data.args.args.outdir, filename)
    if is_overwrite or not os.path.exists(save_path):
        cv2.imwrite(save_path, image)


def save_cadence_frame_and_depth_map_if_active(data: RenderData, frame, image):
    save_cadence_frame(data, frame.i, image)
    if data.args.anim_args.save_depth_maps:
        dm_save_path = os.path.join(data.output_directory, filename_utils.frame_filename(data, frame.i, True))
        data.depth_model.save(dm_save_path, frame.depth)


def load_image(image_path):
    if not os.path.isfile(image_path):
        print(f"File does not exist: {image_path}")
        return None
    return cv2.imread(str(image_path))


def save_and_return_frame(data: RenderData, frame, image):
    save_cadence_frame_and_depth_map_if_active(data, frame, image)
    return image
