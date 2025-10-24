"""Pure functions for 3D transformations and matrix operations.

This module contains transformation-related pure functions extracted from
scripts/deforum_helpers/animation.py, following functional programming principles
with no side effects.
"""

import numpy as np
import cv2
import torch
from functools import reduce
from einops import rearrange
from typing import Tuple, List

# ============================================================================
# TENSOR/IMAGE CONVERSIONS
# ============================================================================


def normalize_image_to_tensor_range(img_array: np.ndarray) -> np.ndarray:
    """Normalize uint8 image [0,255] to tensor range [-1,1].

    Args:
        img_array: Image array in uint8 format

    Returns:
        Normalized array in [-1, 1] range
    """
    return ((img_array.astype(float) / 255.0) * 2) - 1


def denormalize_tensor_to_image_range(tensor_array: np.ndarray) -> np.ndarray:
    """Denormalize tensor range [-1,1] to image range [0,1].

    Args:
        tensor_array: Tensor in [-1, 1] range

    Returns:
        Denormalized array in [0, 1] range
    """
    return ((tensor_array * 0.5) + 0.5).clip(0, 1)


def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    """Convert CV2 image (HWC uint8) to PyTorch tensor (NCHW float16).

    Args:
        sample: CV2 image in HWC format, uint8

    Returns:
        PyTorch tensor in NCHW format, float16
    """
    normalized = normalize_image_to_tensor_range(sample)
    tensor_data = normalized[None].transpose(0, 3, 1, 2).astype(np.float16)
    return torch.from_numpy(tensor_data)


def sample_to_cv2(sample: torch.Tensor, dtype: type = np.uint8) -> np.ndarray:
    """Convert PyTorch tensor (NCHW) to CV2 image (HWC uint8).

    Args:
        sample: PyTorch tensor in NCHW format
        dtype: Output dtype (default: uint8)

    Returns:
        CV2 image in HWC format
    """
    array_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    normalized = denormalize_tensor_to_image_range(array_f32)
    return (normalized * 255).astype(dtype)


# ============================================================================
# ROTATION MATRICES
# ============================================================================


def construct_rotation_matrix_rodrigues(rotation_angles: List[float]) -> np.ndarray:
    """Construct 4x4 homogeneous rotation matrix using Rodrigues formula.

    Args:
        rotation_angles: List of 3 rotation angles [x, y, z] in radians

    Returns:
        4x4 rotation matrix

    Raises:
        ValueError: If rotation_angles is not a list of 3 floats
    """
    if not (isinstance(rotation_angles, list) and len(rotation_angles) == 3):
        raise ValueError("rotation_angles must be list of 3 floats")

    rotation_matrix = np.eye(4, 4)
    cv2.Rodrigues(np.array(rotation_angles), rotation_matrix[0:3, 0:3])
    return rotation_matrix


def create_rotation_matrix_x(angle_rad: float) -> np.ndarray:
    """Create 4x4 rotation matrix around X axis.

    Args:
        angle_rad: Rotation angle in radians

    Returns:
        4x4 rotation matrix
    """
    matrix = np.eye(4, 4)
    sin_a = np.sin(angle_rad)
    cos_a = np.cos(angle_rad)
    matrix[1, 1] = cos_a
    matrix[2, 2] = cos_a
    matrix[1, 2] = -sin_a
    matrix[2, 1] = sin_a
    return matrix


def create_rotation_matrix_y(angle_rad: float) -> np.ndarray:
    """Create 4x4 rotation matrix around Y axis.

    Args:
        angle_rad: Rotation angle in radians

    Returns:
        4x4 rotation matrix
    """
    matrix = np.eye(4, 4)
    sin_a = np.sin(angle_rad)
    cos_a = np.cos(angle_rad)
    matrix[0, 0] = cos_a
    matrix[2, 2] = cos_a
    matrix[0, 2] = sin_a
    matrix[2, 0] = -sin_a
    return matrix


def create_rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """Create 4x4 rotation matrix around Z axis (in-image-plane).

    Args:
        angle_rad: Rotation angle in radians

    Returns:
        4x4 rotation matrix
    """
    matrix = np.eye(4, 4)
    sin_a = np.sin(angle_rad)
    cos_a = np.cos(angle_rad)
    matrix[0, 0] = cos_a
    matrix[1, 1] = cos_a
    matrix[0, 1] = -sin_a
    matrix[1, 0] = sin_a
    return matrix


def get_rotation_matrix_manual(rotation_angles: List[float]) -> np.ndarray:
    """Construct rotation matrix manually using Euler angles.

    See: https://en.wikipedia.org/wiki/Rotation_matrix

    Args:
        rotation_angles: [phi, gamma, theta] in degrees

    Returns:
        4x4 combined rotation matrix
    """
    angles_rad = [np.deg2rad(x) for x in rotation_angles]
    phi = angles_rad[0]  # around X
    gamma = angles_rad[1]  # around Y
    theta = angles_rad[2]  # around Z

    r_phi = create_rotation_matrix_x(phi)
    r_gamma = create_rotation_matrix_y(gamma)
    r_theta = create_rotation_matrix_z(theta)

    return reduce(lambda x, y: np.matmul(x, y), [r_phi, r_gamma, r_theta])


# ============================================================================
# PERSPECTIVE TRANSFORMATION HELPERS
# ============================================================================


def extract_2d_points(pts_3d: np.ndarray) -> np.ndarray:
    """Extract 2D points from 3D perspective transform output.

    Args:
        pts_3d: 3D points from cv2.perspectiveTransform

    Returns:
        2D points array (4, 2)
    """
    pts_2d = pts_3d[0, :]
    return np.array([[pts_2d[i, 0], pts_2d[i, 1]] for i in range(4)])


def get_perspective_transform_points(
    pts_in: np.ndarray, pts_out: np.ndarray, width: int, height: int, side_length: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate input/output points for perspective transform estimation.

    Args:
        pts_in: Input corner points
        pts_out: Output corner points after transformation
        width: Image width
        height: Image height
        side_length: Side length of transformed region

    Returns:
        Tuple of (input_points_2f, output_points_2f) as float32
    """
    pts_in_2d = extract_2d_points(pts_in)
    pts_out_2d = extract_2d_points(pts_out)

    # Center input points
    pin = pts_in_2d + [width / 2.0, height / 2.0]
    # Scale and center output points
    pout = (pts_out_2d + [1.0, 1.0]) * (0.5 * side_length)

    return pin.astype(np.float32), pout.astype(np.float32)


# ============================================================================
# FOV AND PROJECTION MATRICES
# ============================================================================


def calculate_fov_parameters(
    width: int, height: int, fov_deg: float, scale: float
) -> Tuple[float, float, float, float]:
    """Calculate field of view geometry parameters.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        fov_deg: Field of view in degrees
        scale: Scale factor

    Returns:
        Tuple of (side_length, focal_distance, near_plane, far_plane)
    """
    fov_half_rad = np.deg2rad(fov_deg / 2.0)
    diagonal = np.sqrt(width * width + height * height)
    side_length = scale * diagonal / np.cos(fov_half_rad)
    focal_distance = diagonal / (2.0 * np.sin(fov_half_rad))
    near_plane = focal_distance - (diagonal / 2.0)
    far_plane = focal_distance + (diagonal / 2.0)
    return side_length, focal_distance, near_plane, far_plane


def create_translation_matrix_z(z_offset: float) -> np.ndarray:
    """Create 4x4 translation matrix along Z axis.

    Args:
        z_offset: Translation distance along Z

    Returns:
        4x4 translation matrix
    """
    matrix = np.eye(4, 4)
    matrix[2, 3] = z_offset
    return matrix


def create_projection_matrix(fov_half_rad: float, near: float, far: float) -> np.ndarray:
    """Create 4x4 perspective projection matrix.

    Args:
        fov_half_rad: Half field of view in radians
        near: Near clipping plane distance
        far: Far clipping plane distance

    Returns:
        4x4 projection matrix
    """
    matrix = np.eye(4, 4)
    matrix[0, 0] = 1.0 / np.tan(fov_half_rad)
    matrix[1, 1] = matrix[0, 0]
    matrix[2, 2] = -(far + near) / (far - near)
    matrix[2, 3] = -(2.0 * far * near) / (far - near)
    matrix[3, 2] = -1.0
    return matrix


def create_corner_points(width: int, height: int) -> np.ndarray:
    """Create standard corner points array for perspective transform.

    Args:
        width: Image width
        height: Image height

    Returns:
        Corner points array (1, 4, 3) for cv2.perspectiveTransform
    """
    return np.array(
        [
            [
                [-width / 2.0, height / 2.0, 0.0],
                [width / 2.0, height / 2.0, 0.0],
                [width / 2.0, -height / 2.0, 0.0],
                [-width / 2.0, -height / 2.0, 0.0],
            ]
        ]
    )


# ============================================================================
# COMPLETE WARP MATRIX
# ============================================================================


def warpMatrix(
    W: int, H: int, theta: float, phi: float, gamma: float, scale: float, fV: float
) -> Tuple[np.ndarray, float]:
    """Calculate perspective warp matrix for 3D rotation and projection.

    Args:
        W: Image width
        H: Image height
        theta: Rotation around Z (in-plane) in degrees
        phi: Rotation around X in degrees
        gamma: Rotation around Y in degrees
        scale: Scale factor
        fV: Field of view in degrees

    Returns:
        Tuple of (M33, side_length) where M33 is 3x3 perspective transform
    """
    # Calculate FOV parameters
    side_length, focal_distance, near, far = calculate_fov_parameters(W, H, fV, scale)
    fov_half_rad = np.deg2rad(fV / 2.0)

    # Build transformation matrices
    T = create_translation_matrix_z(-focal_distance)
    R = get_rotation_matrix_manual([phi, gamma, theta])
    P = create_projection_matrix(fov_half_rad, near, far)

    # Combine transformations: Projection * Translation * Rotation
    F = reduce(lambda x, y: np.matmul(x, y), [P, T, R])

    # Create corner points and apply perspective transform
    pts_in = create_corner_points(W, H)
    pts_out = cv2.perspectiveTransform(pts_in, F)

    # Get points for final transform estimation
    pts_in_2f, pts_out_2f = get_perspective_transform_points(pts_in, pts_out, W, H, side_length)

    # Verify float32 type (required by OpenCV)
    assert pts_in_2f.dtype == np.float32
    assert pts_out_2f.dtype == np.float32

    # Calculate final 3x3 perspective transform matrix
    M33 = cv2.getPerspectiveTransform(pts_in_2f, pts_out_2f)

    return M33, side_length
