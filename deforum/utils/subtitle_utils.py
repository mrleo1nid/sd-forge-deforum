"""Pure functions for subtitle generation and time formatting.

This module contains subtitle-related pure functions extracted from
scripts/deforum_helpers/subtitle_handler.py, following functional
programming principles with no side effects.
"""

from decimal import Decimal, getcontext


def time_to_srt_format(seconds: float | Decimal) -> str:
    """Convert seconds to SRT subtitle format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds (float or Decimal)

    Returns:
        Formatted time string (e.g., "00:01:23,456")

    Examples:
        >>> time_to_srt_format(0)
        '00:00:00,000'
        >>> time_to_srt_format(83.5)
        '00:01:23,500'
        >>> time_to_srt_format(3661.123)
        '01:01:01,123'
    """
    hours, remainder = divmod(float(seconds), 3600)
    minutes, remainder = divmod(remainder, 60)
    seconds_part, milliseconds = divmod(remainder, 1)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds_part):02},{int(milliseconds * 1000):03}"


def calculate_frame_duration(fps: float, precision: int = 20) -> Decimal:
    """Calculate duration of a single frame in seconds.

    Args:
        fps: Frames per second
        precision: Decimal precision for calculation (default: 20)

    Returns:
        Frame duration as Decimal (1/fps)

    Examples:
        >>> duration = calculate_frame_duration(30)
        >>> float(duration)
        0.03333333333333333333
        >>> duration = calculate_frame_duration(60)
        >>> float(duration)
        0.01666666666666666667
    """
    getcontext().prec = precision
    return Decimal(1) / Decimal(fps)


def frame_time(frame_number: int, frame_duration: Decimal) -> Decimal:
    """Calculate timestamp for a specific frame.

    Args:
        frame_number: Frame index (0-based)
        frame_duration: Duration of one frame in seconds

    Returns:
        Time in seconds for this frame

    Examples:
        >>> duration = calculate_frame_duration(30)
        >>> time = frame_time(0, duration)
        >>> float(time)
        0.0
        >>> time = frame_time(30, duration)
        >>> float(time)
        1.0
    """
    return Decimal(frame_number) * frame_duration


def format_subtitle_value(param_value) -> str:
    """Format parameter value for subtitle display.

    Converts values to strings with appropriate formatting:
    - Floats that are whole numbers: "3" (not "3.0")
    - Floats with decimals: "3.140" (3 decimal places)
    - Other types: Direct string conversion

    Args:
        param_value: Value to format (any type)

    Returns:
        Formatted string representation

    Examples:
        >>> format_subtitle_value(3.0)
        '3'
        >>> format_subtitle_value(3.14159)
        '3.142'
        >>> format_subtitle_value(42)
        '42'
        >>> format_subtitle_value("test")
        'test'
    """
    is_float = isinstance(param_value, float)
    if is_float and param_value == int(param_value):
        return str(int(param_value))
    elif is_float and not param_value.is_integer():
        return f"{param_value:.3f}"
    else:
        return f"{param_value}"


# Parameter display name mapping
# Maps internal parameter names to user-friendly display names
SUBTITLE_PARAM_NAMES = {
    "angle": {"backend": "angle_series", "user": "Angle", "print": "Angle"},
    "transform_center_x": {"backend": "transform_center_x_series", "user": "Trans Center X", "print": "Tr.C.X"},
    "transform_center_y": {"backend": "transform_center_y_series", "user": "Trans Center Y", "print": "Tr.C.Y"},
    "zoom": {"backend": "zoom_series", "user": "Zoom", "print": "Zoom"},
    "translation_x": {"backend": "translation_x_series", "user": "Trans X", "print": "TrX"},
    "translation_y": {"backend": "translation_y_series", "user": "Trans Y", "print": "TrY"},
    "translation_z": {"backend": "translation_z_series", "user": "Trans Z", "print": "TrZ"},
    "rotation_3d_x": {"backend": "rotation_3d_x_series", "user": "Rot 3D X", "print": "RotX"},
    "rotation_3d_y": {"backend": "rotation_3d_y_series", "user": "Rot 3D Y", "print": "RotY"},
    "rotation_3d_z": {"backend": "rotation_3d_z_series", "user": "Rot 3D Z", "print": "RotZ"},
    "perspective_flip_theta": {"backend": "perspective_flip_theta_series", "user": "Per Fl Theta", "print": "PerFlT"},
    "perspective_flip_phi": {"backend": "perspective_flip_phi_series", "user": "Per Fl Phi", "print": "PerFlP"},
    "perspective_flip_gamma": {"backend": "perspective_flip_gamma_series", "user": "Per Fl Gamma", "print": "PerFlG"},
    "perspective_flip_fv": {"backend": "perspective_flip_fv_series", "user": "Per Fl FV", "print": "PerFlFV"},
    "noise_schedule": {"backend": "noise_schedule_series", "user": "Noise Sch", "print": "Noise"},
    "strength_schedule": {"backend": "strength_schedule_series", "user": "Str Sch", "print": "StrSch"},
    "keyframe_strength_schedule": {"backend": "keyframe_strength_schedule_series", "user": "Kfr Str Sch", "print": "KfrStrSch"},
    "contrast_schedule": {"backend": "contrast_schedule_series", "user": "Contrast Sch", "print": "CtrstSch"},
    "cfg_scale_schedule": {"backend": "cfg_scale_schedule_series", "user": "CFG Sch", "print": "CFGSch"},
    "distilled_cfg_scale_schedule": {"backend": "distilled_cfg_scale_schedule_series", "user": "Dist. CFG Sch", "print": "DistCFGSch"},
    "subseed_schedule": {"backend": "subseed_schedule_series", "user": "Subseed Sch", "print": "SubSSch"},
    "subseed_strength_schedule": {"backend": "subseed_strength_schedule_series", "user": "Subseed Str Sch", "print": "SubSStrSch"},
    "checkpoint_schedule": {"backend": "checkpoint_schedule_series", "user": "Ckpt Sch", "print": "CkptSch"},
    "steps_schedule": {"backend": "steps_schedule_series", "user": "Steps Sch", "print": "StepsSch"},
    "seed_schedule": {"backend": "seed_schedule_series", "user": "Seed Sch", "print": "SeedSch"},
    "sampler_schedule": {"backend": "sampler_schedule_series", "user": "Sampler Sch", "print": "SamplerSchedule"},
    "scheduler_schedule": {"backend": "scheduler_schedule_series", "user": "Scheduler Sch", "print": "SchedulerSchedule"},
    "clipskip_schedule": {"backend": "clipskip_schedule_series", "user": "Clipskip Sch", "print": "ClipskipSchedule"},
    "noise_multiplier_schedule": {"backend": "noise_multiplier_schedule_series", "user": "Noise Multp Sch", "print": "NoiseMultiplierSchedule"},
    "mask_schedule": {"backend": "mask_schedule_series", "user": "Mask Sch", "print": "MaskSchedule"},
    "noise_mask_schedule": {"backend": "noise_mask_schedule_series", "user": "Noise Mask Sch", "print": "NoiseMaskSchedule"},
    "amount_schedule": {"backend": "amount_schedule_series", "user": "Ant.Blr Amount Sch", "print": "AmountSchedule"},
    "kernel_schedule": {"backend": "kernel_schedule_series", "user": "Ant.Blr Kernel Sch", "print": "KernelSchedule"},
    "sigma_schedule": {"backend": "sigma_schedule_series", "user": "Ant.Blr Sigma Sch", "print": "SigmaSchedule"},
    "threshold_schedule": {"backend": "threshold_schedule_series", "user": "Ant.Blr Threshold Sch", "print": "ThresholdSchedule"},
    "aspect_ratio_schedule": {"backend": "aspect_ratio_series", "user": "Aspect Ratio Sch", "print": "AspectRatioSchedule"},
    "fov_schedule": {"backend": "fov_series", "user": "FOV Sch", "print": "FieldOfViewSchedule"},
    "near_schedule": {"backend": "near_series", "user": "Near Sch", "print": "NearSchedule"},
    "cadence_flow_factor_schedule": {"backend": "cadence_flow_factor_schedule_series", "user": "Cadence Flow Factor Sch", "print": "CadenceFlowFactorSchedule"},
    "redo_flow_factor_schedule": {"backend": "redo_flow_factor_schedule_series", "user": "Redo Flow Factor Sch", "print": "RedoFlowFactorSchedule"},
    "far_schedule": {"backend": "far_series", "user": "Far Sch", "print": "FarSchedule"},
}


def get_user_param_names() -> list[str]:
    """Get list of user-friendly parameter names for subtitle display.

    Returns:
        List of user-facing parameter names including "Prompt"

    Examples:
        >>> names = get_user_param_names()
        >>> "Angle" in names
        True
        >>> "Prompt" in names
        True
        >>> len(names) > 30
        True
    """
    items = [v["user"] for v in SUBTITLE_PARAM_NAMES.values()]
    items.append("Prompt")
    return items
