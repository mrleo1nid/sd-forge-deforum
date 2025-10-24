# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

"""
Optical flow utility functions extracted from hybrid_video.py.
These are used by optical flow cadence (non-hybrid feature).
"""

import cv2
import numpy as np

# Import pure functions from refactored utils modules
from deforum.utils.image_geometry_utils import center_crop, extend_with_grid
from deforum.utils.optical_flow_utils import convert_relative_flow_to_absolute


def remap(img, flow):
    """Remap image using flow field with border handling."""
    border_mode = cv2.BORDER_REFLECT_101
    h, w = img.shape[:2]
    displacement = int(h * 0.25), int(w * 0.25)
    larger_img = cv2.copyMakeBorder(img, displacement[0], displacement[0], displacement[1], displacement[1], border_mode)
    lh, lw = larger_img.shape[:2]
    larger_flow = extend_flow(flow, lw, lh)
    remapped_img = cv2.remap(larger_img, larger_flow, None, cv2.INTER_LINEAR, border_mode)
    output_img = center_crop_image(remapped_img, w, h)
    return output_img


def center_crop_image(img, w, h):
    """Crop image to specified width and height from center.

    Backward compatibility wrapper for center_crop from utils.
    """
    return center_crop(img, w, h)


def extend_flow(flow, w, h):
    """Extend flow field to specified dimensions.

    Backward compatibility wrapper for extend_with_grid from utils.
    """
    return extend_with_grid(flow, w, h)


def image_transform_optical_flow(img, flow, flow_factor):
    """Transform image using optical flow with optional flow factor scaling."""
    # if flow factor not normal, calculate flow factor
    if flow_factor != 1:
        flow = flow * flow_factor
    # flow is reversed, so you need to reverse it:
    flow = -flow
    h, w = img.shape[:2]
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:,np.newaxis]
    return remap(img, flow)


def abs_flow_to_rel_flow(flow, width, height):
    """Convert absolute flow to relative flow."""
    fx, fy = flow[:,:,0], flow[:,:,1]
    max_flow_x = np.max(np.abs(fx))
    max_flow_y = np.max(np.abs(fy))
    max_flow = max(max_flow_x, max_flow_y)

    rel_fx = fx / (max_flow * width)
    rel_fy = fy / (max_flow * height)
    return np.dstack((rel_fx, rel_fy))


def rel_flow_to_abs_flow(rel_flow, width, height):
    """Convert relative flow to absolute flow.

    Backward compatibility wrapper for convert_relative_flow_to_absolute from utils.
    """
    return convert_relative_flow_to_absolute(rel_flow, width, height)


def get_flow_from_images(i1, i2, method, raft_model, prev_flow=None):
    """
    Get optical flow between two images using RAFT method.

    Args:
        i1: First image (PIL or numpy)
        i2: Second image (PIL or numpy)
        method: Flow method (only 'RAFT' supported)
        raft_model: RAFT model instance
        prev_flow: Unused (kept for backwards compatibility)

    Returns:
        Flow array with shape (H, W, 2)
    """
    if method == "RAFT":
        if raft_model is None:
            raise Exception("RAFT Model not provided to get_flow_from_images function, cannot continue.")
        return get_flow_from_images_RAFT(i1, i2, raft_model)
    # Only RAFT is supported
    raise RuntimeError(f"Invalid flow method name: '{method}'. Only 'RAFT' is supported.")


def get_flow_from_images_RAFT(i1, i2, raft_model):
    """Get optical flow using RAFT model."""
    flow = raft_model.predict(i1, i2)
    return flow


def draw_flow_arrows(depth_image, flow, step=20, max_arrow_length=30, arrow_color=(0, 255, 0), min_magnitude=0.5):
    """Draw optical flow as green arrows on depth preview with normalized lengths.

    Args:
        depth_image: Grayscale depth map (H, W) or (H, W, 1)
        flow: Optical flow array (H, W, 2)
        step: Spacing between arrows in pixels (default 20)
        max_arrow_length: Maximum arrow length in pixels (default 30)
        arrow_color: BGR color tuple for arrows (default green)
        min_magnitude: Minimum flow magnitude to draw as fraction of max (default 0.5 = 50%)

    Returns:
        RGB image with flow arrows overlaid
    """
    # Convert grayscale depth to RGB for colored arrows
    if len(depth_image.shape) == 2:
        vis = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
    elif depth_image.shape[2] == 1:
        vis = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
    else:
        vis = depth_image.copy()

    h, w = vis.shape[:2]

    # Calculate flow magnitudes for normalization
    flow_mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    max_mag = np.max(flow_mag)

    # Avoid division by zero
    if max_mag < 1e-6:
        return vis

    # Draw arrows on subsampled grid
    for y in range(0, h, step):
        for x in range(0, w, step):
            if y < flow.shape[0] and x < flow.shape[1]:
                fx, fy = flow[y, x]
                mag = np.sqrt(fx**2 + fy**2)

                # Only draw if flow magnitude > threshold (as fraction of max)
                if mag > max_mag * min_magnitude:
                    # Normalize flow to max_arrow_length
                    # Longest arrows will be max_arrow_length pixels
                    scale = max_arrow_length / max_mag

                    # Calculate arrow endpoint
                    x_end = int(x + fx * scale)
                    y_end = int(y + fy * scale)

                    # Clamp to image bounds
                    x_end = max(0, min(w - 1, x_end))
                    y_end = max(0, min(h - 1, y_end))

                    # Vary arrow thickness based on magnitude (1-2 pixels)
                    thickness = 1 if mag < max_mag * 0.7 else 2

                    # Draw arrow
                    cv2.arrowedLine(vis, (x, y), (x_end, y_end),
                                    arrow_color, thickness, tipLength=0.3)
    return vis
