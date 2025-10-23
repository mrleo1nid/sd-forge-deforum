from cv2.typing import MatLike

from .call.anim import call_anim_frame_warp
from ...optical_flow_utils import (get_flow_from_images, image_transform_optical_flow, rel_flow_to_abs_flow)


def advance_optical_flow_cadence_before_animation_warping(data, last_frame, tween_frame, prev_image, image) -> MatLike:
    is_with_flow = data.is_3d_or_2d_with_optical_flow()
    if is_with_flow and _is_do_flow(data, tween_frame, last_frame.i, prev_image, image):
        method = data.args.anim_args.optical_flow_cadence  # string containing the flow method (e.g. "RAFT").
        flow = get_flow_from_images(prev_image, image, method, data.animation_mode.raft_model)
        # Store per-tween flow for this tween's interpolation position
        tween_frame.cadence_flow = flow / len(last_frame.tweens)
        # Apply flow only ONCE (removed double application bug)
        advanced_image = _advance_optical_flow(tween_frame, image)
        return advanced_image
    return image


def advance(data, i, image, depth):
    if depth is not None:
        warped_image, _ = call_anim_frame_warp(data, i, image, depth)
        return warped_image
    else:
        return image


def do_optical_flow_cadence_after_animation_warping(data, tween_frame, prev_image, image):
    # After 3D warping is applied, optionally blend with previous frame
    # based on tween interpolation value (0.0 = prev, 1.0 = current)
    if prev_image is not None and tween_frame.value < 1.0:
        return prev_image * (1.0 - tween_frame.value) + image * tween_frame.value
    return image


def _advance_optical_flow(tween_step, image, flow_factor: int = 1):
    flow = tween_step.cadence_flow * -1
    return image_transform_optical_flow(image, flow, flow_factor)


def _is_do_flow(data, tween_frame, start_i, prev_image, image):
    has_tween_schedule = data.animation_keys.deform_keys.strength_schedule_series[start_i] > 0
    has_images = prev_image is not None and image is not None
    has_step_and_images = tween_frame.cadence_flow is None and has_images
    return has_tween_schedule and has_step_and_images and data.animation_mode.is_raft_active()
