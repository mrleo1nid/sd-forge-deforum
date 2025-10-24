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

import numpy as np
import torch

# Import pure functions from refactored utils module
from deforum.utils.math.core import (
    rotation_matrix,
    rotate_camera_towards_depth,
)

# Re-export for backward compatibility
__all__ = [
    'rotation_matrix',
    'rotate_camera_towards_depth',
]

# reallybigname - auto-navigation functions in progress...
# usage:
# if auto_rotation:
#    rot_mat = rotate_camera_towards_depth(depth_tensor, auto_rotation_steps, w, h, fov_deg, auto_rotation_depth_target)

# rotate_camera_towards_depth imported from deforum.utils.math.core
# rotation_matrix imported from deforum.utils.math.core
