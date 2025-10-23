import os

import PIL
import cv2
import numpy as np
from PIL import Image
from cv2.typing import MatLike

from . import filename_utils
from ..data.render_data import RenderData


def bgr_to_rgb(bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


def numpy_to_pil(np_image: MatLike) -> Image.Image:
    return Image.fromarray(bgr_to_rgb(np_image))


def pil_to_numpy(pil_image: Image.Image) -> MatLike:
    return np.array(pil_image)


def save_cadence_frame(data: RenderData, i: int, image: MatLike, is_overwrite: bool = True):
    filename = filename_utils.frame_filename(data, i)
    save_path: str = os.path.join(data.args.args.outdir, filename)
    if is_overwrite or not os.path.exists(save_path):
        cv2.imwrite(save_path, image)

        # Also copy to frame-preview.png for UI live preview
        preview_path = os.path.join(data.args.args.outdir, "frame-preview.png")
        cv2.imwrite(preview_path, image)


def save_cadence_frame_and_depth_map_if_active(data: RenderData, frame, image):
    save_cadence_frame(data, frame.i, image)
    if data.args.anim_args.save_depth_maps:
        dm_save_path = os.path.join(data.output_directory, filename_utils.frame_filename(data, frame.i, True))
        # Save the depth map first
        data.depth_model.save(dm_save_path, frame.depth)

        # Always create a depth preview for the UI
        depth_preview_path = os.path.join(data.args.args.outdir, "depth-raft-preview.png")

        # Check if we should add flow arrows to depth preview
        show_flow_arrows = getattr(data.args.anim_args, 'show_flow_arrows', False)
        if show_flow_arrows and hasattr(frame, 'cadence_flow') and frame.cadence_flow is not None:
            # Import here to avoid circular dependency
            from ...optical_flow_utils import draw_flow_arrows
            import cv2

            # Load the just-saved depth map
            depth_image = cv2.imread(dm_save_path, cv2.IMREAD_UNCHANGED)
            if depth_image is not None:
                # Draw flow arrows on depth map
                depth_with_flow = draw_flow_arrows(depth_image, frame.cadence_flow)
                # Save with flow visualization (per-frame file)
                flow_save_path = dm_save_path.replace('.png', '_flow.png')
                cv2.imwrite(flow_save_path, depth_with_flow)
                # Also update the live preview
                cv2.imwrite(depth_preview_path, depth_with_flow)
        else:
            # No flow arrows - just copy the plain depth map to preview
            import cv2
            depth_image = cv2.imread(dm_save_path, cv2.IMREAD_UNCHANGED)
            if depth_image is not None:
                cv2.imwrite(depth_preview_path, depth_image)


def load_image(image_path):
    if not os.path.isfile(image_path):
        print(f"File does not exist: {image_path}")
        return None
    return cv2.imread(str(image_path))


def save_and_return_frame(data: RenderData, frame, image):
    save_cadence_frame_and_depth_map_if_active(data, frame, image)
    return image


def is_PIL(image):
    return type(image) is PIL.Image.Image
