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


def rel_flow_to_abs_flow(rel_flow, width, height):
    """Convert relative flow to absolute flow.

    Backward compatibility wrapper for convert_relative_flow_to_absolute from utils.
    """
    return convert_relative_flow_to_absolute(rel_flow, width, height)


def get_flow_from_images(i1, i2, method, raft_model, prev_flow=None):
    """
    Get optical flow between two images using specified method.
    Supports: RAFT, DIS (Medium/Fine), Farneback.
    """
    if method == "RAFT":
        if raft_model is None:
            raise Exception("RAFT Model not provided to get_flow_from_images function, cannot continue.")
        return get_flow_from_images_RAFT(i1, i2, raft_model)
    elif method == "DIS Medium":
        return get_flow_from_images_DIS(i1, i2, 'medium', prev_flow)
    elif method == "DIS Fine":
        return get_flow_from_images_DIS(i1, i2, 'fine', prev_flow)
    elif method == "Farneback":
        return get_flow_from_images_Farneback(i1, i2, last_flow=prev_flow)
    # if we reached this point, something went wrong. raise an error:
    raise RuntimeError(f"Invalid flow method name: '{method}'")


def get_flow_from_images_RAFT(i1, i2, raft_model):
    """Get optical flow using RAFT model."""
    flow = raft_model.predict(i1, i2)
    return flow


def get_flow_from_images_DIS(i1, i2, preset, prev_flow):
    """Get optical flow using DIS (Dense Inverse Search) method."""
    # DIS PRESETS CHART KEY: finest scale, grad desc its, patch size
    # DIS_MEDIUM: 1, 25, 8 | DIS_FAST: 2, 16, 8 | DIS_ULTRAFAST: 2, 12, 8
    if preset == 'medium':
        preset_code = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    elif preset == 'fast':
        preset_code = cv2.DISOPTICAL_FLOW_PRESET_FAST
    elif preset == 'ultrafast':
        preset_code = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
    elif preset in ['slow','fine']:
        preset_code = None

    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(preset_code)

    # custom presets
    if preset == 'slow':
        dis.setGradientDescentIterations(192)
        dis.setFinestScale(1)
        dis.setPatchSize(8)
        dis.setPatchStride(4)
    if preset == 'fine':
        dis.setGradientDescentIterations(192)
        dis.setFinestScale(0)
        dis.setPatchSize(8)
        dis.setPatchStride(4)

    return dis.calc(i1, i2, prev_flow)


def get_flow_from_images_Farneback(i1, i2, preset="normal", last_flow=None,
                                    pyr_scale=0.5, levels=3, winsize=15,
                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0):
    """Get optical flow using Farneback method."""
    flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN  # Specify the operation flags
    pyr_scale = 0.5  # The image scale (<1) to build pyramids for each image

    if preset == "fine":
        levels = 13       # The number of pyramid layers
        winsize = 77      # The averaging window size
        iterations = 13   # The number of iterations at each pyramid level
        poly_n = 15       # The size of the pixel neighborhood
        poly_sigma = 0.8  # The standard deviation of the Gaussian
    else:  # "normal"
        levels = 5
        winsize = 21
        iterations = 5
        poly_n = 7
        poly_sigma = 1.2

    i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
    i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
    flags = 0  # flags = cv2.OPTFLOW_USE_INITIAL_FLOW
    flow = cv2.calcOpticalFlowFarneback(i1, i2, last_flow, pyr_scale, levels,
                                        winsize, iterations, poly_n, poly_sigma, flags)
    return flow
