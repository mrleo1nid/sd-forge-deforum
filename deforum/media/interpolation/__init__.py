"""Frame interpolation for Deforum.

Provides frame interpolation capabilities for smooth transitions between
keyframes using various algorithms (RIFE, FILM, etc.).
"""

from .frame_interpolation import (
    gradio_f_interp_get_fps_and_fcount,
    process_interp_vid_upload_logic,
    process_video_interpolation,
    process_interp_pics_upload_logic,
)

# Re-export utility function from utils
from deforum.utils.interpolation_utils import clean_folder_name

__all__ = [
    "gradio_f_interp_get_fps_and_fcount",
    "process_interp_vid_upload_logic",
    "process_video_interpolation",
    "process_interp_pics_upload_logic",
    "clean_folder_name",
]
