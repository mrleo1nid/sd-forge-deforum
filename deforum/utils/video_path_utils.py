"""Pure functions for video and image path generation.

This module contains path generation functions extracted from
scripts/deforum_helpers/video_audio_utilities.py, following functional
programming principles with no side effects.
"""

import os


def get_next_frame_path(
    outdir: str, video_name: str, frame_idx: int, is_mask: bool = False
) -> str:
    """Generate path to next frame image based on video name and frame index.

    Constructs the full path to a frame image file within the appropriate
    subdirectory (inputframes or maskframes) using the video's base name
    and zero-padded frame index.

    Args:
        outdir: Output directory containing inputframes/maskframes subdirectories
        video_name: Base name extracted from video path (without extension)
        frame_idx: Frame index (0-based)
        is_mask: If True, use maskframes directory; otherwise use inputframes

    Returns:
        Full path to the frame image file (JPEG format with 9-digit zero-padded index)

    Examples:
        >>> get_next_frame_path('/output', 'video', 0, False)
        '/output/inputframes/video000000000.jpg'
        >>> get_next_frame_path('/output', 'myvid', 42, True)
        '/output/maskframes/myvid000000042.jpg'
        >>> get_next_frame_path('/data', 'test', 999, False)
        '/data/inputframes/test000000999.jpg'
    """
    frame_subdir = "maskframes" if is_mask else "inputframes"
    filename = f"{video_name}{frame_idx:09d}.jpg"
    return os.path.join(outdir, frame_subdir, filename)


def get_output_video_path(input_image_path: str) -> str:
    """Generate unique output video path from input image path pattern.

    Creates an output MP4 path in the same directory as the input images,
    using the parent folder's name. If the file already exists, appends
    a numeric suffix (_1, _2, etc.) to ensure uniqueness.

    Args:
        input_image_path: Path to input image (may contain printf-style patterns)

    Returns:
        Unique MP4 output path that doesn't conflict with existing files

    Examples:
        >>> # If /data/frames/ contains no videos:
        >>> get_output_video_path('/data/frames/image.png')
        '/data/frames/frames.mp4'

        >>> # If /data/myrun/ already has myrun.mp4:
        >>> get_output_video_path('/data/myrun/img001.jpg')
        '/data/myrun/myrun_1.mp4'

        >>> # If /videos/test/ has test.mp4 and test_1.mp4:
        >>> get_output_video_path('/videos/test/frame%05d.png')
        '/videos/test/test_2.mp4'
    """
    # Get directory and folder name from input path
    dir_name = os.path.dirname(input_image_path)
    folder_name = os.path.basename(dir_name)

    # Start with base output path
    output_path = os.path.join(dir_name, f"{folder_name}.mp4")

    # If file exists, append numeric suffix until we find unique name
    suffix = 1
    while os.path.exists(output_path):
        output_path = os.path.join(dir_name, f"{folder_name}_{suffix}.mp4")
        suffix += 1

    return output_path
