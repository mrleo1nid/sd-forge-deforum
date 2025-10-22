"""Pure functions for optical flow consistency checking.

This module contains optical flow-related pure functions extracted from
scripts/deforum_helpers/consistency_check.py, following functional programming
principles with no side effects.

Original implementation taken from https://github.com/Sxela/flow_tools (GNU GPL Licensed)
and adapted for Deforum.
"""

import numpy as np


def make_consistency(
    flow1: np.ndarray, flow2: np.ndarray, edges_unreliable: bool = False
) -> np.ndarray:
    """Calculate optical flow consistency map between forward and backward flows.

    Checks consistency of forward flow via backward flow using bilinear interpolation
    and motion edge detection. Based on the algorithm from:
    https://github.com/manuelruder/artistic-videos/blob/master/consistencyChecker/consistencyChecker.cpp

    Args:
        flow1: Forward optical flow (H x W x 2) where channels are (u, v)
        flow2: Backward optical flow (H x W x 2) where channels are (u, v)
        edges_unreliable: If True, mark edges as unreliable in output

    Returns:
        Reliability map (H x W x 3) where:
            - Channel 0: Occlusion detection (-0.75 for occluded, 1.0 for reliable)
            - Channel 1: Edge detection (0 for edges, 1.0 otherwise)
            - Channel 2: Motion edge detection (0 for motion edges, 1.0 otherwise)

    Examples:
        >>> import numpy as np
        >>> flow_fwd = np.random.randn(100, 100, 2)
        >>> flow_bwd = -flow_fwd  # Approximate inverse
        >>> consistency = make_consistency(flow_fwd, flow_bwd)
        >>> consistency.shape
        (100, 100, 3)
    """
    # Flip flow channels for processing
    flow1 = np.flip(flow1, axis=2)
    flow2 = np.flip(flow2, axis=2)
    h, w, _ = flow1.shape

    # Get grid of coordinates for each pixel
    orig_coord = np.flip(np.mgrid[:w, :h], 0).T

    # Find where flow1 maps each pixel
    warp_coord = orig_coord + flow1

    # Clip the coordinates in bounds and round down
    warp_coord_inbound = np.zeros_like(warp_coord)
    warp_coord_inbound[..., 0] = np.clip(warp_coord[..., 0], 0, h - 2)
    warp_coord_inbound[..., 1] = np.clip(warp_coord[..., 1], 0, w - 2)
    warp_coord_floor = np.floor(warp_coord_inbound).astype(int)

    # Bilinear interpolation of flow2 values around the point mapped to by flow1
    alpha = warp_coord_inbound - warp_coord_floor
    flow2_00 = flow2[warp_coord_floor[..., 0], warp_coord_floor[..., 1]]
    flow2_01 = flow2[warp_coord_floor[..., 0], warp_coord_floor[..., 1] + 1]
    flow2_10 = flow2[warp_coord_floor[..., 0] + 1, warp_coord_floor[..., 1]]
    flow2_11 = flow2[warp_coord_floor[..., 0] + 1, warp_coord_floor[..., 1] + 1]
    flow2_0_blend = (1 - alpha[..., 1, None]) * flow2_00 + alpha[..., 1, None] * flow2_01
    flow2_1_blend = (1 - alpha[..., 1, None]) * flow2_10 + alpha[..., 1, None] * flow2_11
    warp_coord_flow2 = (
        (1 - alpha[..., 0, None]) * flow2_0_blend + alpha[..., 0, None] * flow2_1_blend
    )

    # Coordinates that flow2 remaps each flow1-mapped pixel to
    rewarp_coord = warp_coord + warp_coord_flow2

    # Detect occlusions: where position difference after flow1 and flow2 is large
    squared_diff = np.sum((rewarp_coord - orig_coord) ** 2, axis=2)
    threshold = 0.01 * np.sum(warp_coord_flow2**2 + flow1**2, axis=2) + 0.5

    reliable_flow = np.ones((squared_diff.shape[0], squared_diff.shape[1], 3))
    reliable_flow[..., 0] = np.where(squared_diff >= threshold, -0.75, 1)

    # Mark areas mapping outside frame as unreliable (if requested)
    if edges_unreliable:
        reliable_flow[..., 1] = np.where(
            np.logical_or.reduce(
                (
                    warp_coord[..., 0] < 0,
                    warp_coord[..., 1] < 0,
                    warp_coord[..., 0] >= h - 1,
                    warp_coord[..., 1] >= w - 1,
                )
            ),
            0,
            reliable_flow[..., 1],
        )

    # Detect motion edges: large changes in flow derivative => edge of moving object
    dx = np.diff(flow1, axis=1, append=0)
    dy = np.diff(flow1, axis=0, append=0)
    motion_edge = np.sum(dx**2 + dy**2, axis=2)
    motion_threshold = 0.01 * np.sum(flow1**2, axis=2) + 0.002
    reliable_flow[..., 2] = np.where(
        np.logical_and(motion_edge > motion_threshold, reliable_flow[..., 2] != -0.75),
        0,
        reliable_flow[..., 2],
    )

    return reliable_flow
