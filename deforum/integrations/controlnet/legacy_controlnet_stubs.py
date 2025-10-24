"""
Stub functions replacing legacy ControlNet integration.

Legacy ControlNet has been removed as it was not working with the current
codebase. These stubs exist to prevent import errors in code that hasn't
been fully migrated yet. New code should use Flux ControlNet V2 instead.
"""

# Constants
num_of_models = 0  # No legacy ControlNet models


def find_controlnet():
    """Stub: Legacy ControlNet is not available."""
    return None


def controlnet_infotext():
    """Stub: No infotext for legacy ControlNet."""
    return ""


def is_controlnet_enabled(controlnet_args):
    """Stub: Legacy ControlNet is always disabled."""
    return False


def setup_controlnet_ui_raw():
    """Stub: No UI for legacy ControlNet."""
    return []


def setup_controlnet_ui():
    """Stub: No UI for legacy ControlNet."""
    return {}


def controlnet_component_names():
    """Stub: No component names for legacy ControlNet."""
    return []


def get_controlnet_script_args(args, anim_args, controlnet_args, root, parseq_adapter, frame_idx=0):
    """Stub: No script args for legacy ControlNet."""
    return []


def find_controlnet_script(p):
    """Stub: Legacy ControlNet script is not available."""
    return None


def process_controlnet_input_frames(args, anim_args, controlnet_args, input_path, is_mask, outdir_suffix, id):
    """Stub: No processing for legacy ControlNet input frames."""
    pass


def unpack_controlnet_vids(args, anim_args, controlnet_args):
    """Stub: No video unpacking for legacy ControlNet."""
    pass
