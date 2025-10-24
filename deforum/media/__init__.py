"""Media handling module for Deforum.

Provides image/video I/O, upscaling, and interpolation capabilities.
"""

from .load_images import (
    load_img,
    load_image,
    get_mask,
    get_mask_from_file,
    # Add other load functions as needed
)
from .save_images import (
    save_image,
    # Add other save functions as needed
)
from .upscaling import (
    process_ncnn_upscale_vid_upload_logic,
    process_ncnn_video_upscaling,
    make_upscale_v2,
)
from .video_audio_utilities import (
    get_frame_name,
    get_next_frame,
    render_preview,
    download_audio,
    # Add other video/audio functions as needed
)

__all__ = [
    # Load
    "load_img",
    "load_image",
    "get_mask",
    "get_mask_from_file",
    # Save
    "save_image",
    # Upscaling
    "process_ncnn_upscale_vid_upload_logic",
    "process_ncnn_video_upscaling",
    "make_upscale_v2",
    # Video/Audio
    "get_frame_name",
    "get_next_frame",
    "render_preview",
    "download_audio",
]
