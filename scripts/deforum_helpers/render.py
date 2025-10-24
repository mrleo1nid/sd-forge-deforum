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

import os
import pandas as pd
import cv2
import numpy as np
import numexpr
import gc
import random
import PIL
import time
from PIL import Image, ImageOps
from .generate import generate, isJson
from .noise import add_noise
from .animation import anim_frame_warp
from .animation_key_frames import DeformAnimKeys, LooperAnimKeys
from .video_audio_utilities import get_frame_name, get_next_frame, render_preview
from deforum.depth import DepthModel
from .colors import maintain_colors
from deforum.integrations.parseq import ParseqAdapter
from .seed import next_seed
from .image_sharpening import unsharp_mask
from .load_images import get_mask, load_img, load_image, get_mask_from_file
# Hybrid video removed - was: from .hybrid_video import (...)
from .save_images import save_image
from .composable_masks import compose_mask_with_check
from deforum.config.settings import save_settings_from_animation_run
from .deforum_controlnet import unpack_controlnet_vids, is_controlnet_enabled
from .subtitle_handler import init_srt_file, write_frame_subtitle, format_animation_params
from .resume import get_resume_vars
from .masks import do_overlay_mask
from .prompt import prepare_prompt
from modules.shared import opts, cmd_opts, state, sd_model
from modules import lowvram, devices, sd_hijack
from .rendering import experimental_core
from .rendering.util import log_utils
from deforum.integrations.raft import RAFT

from deforum_api import JobStatusTracker


def render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):
    # Pre-download soundtrack if specified
    if video_args.add_soundtrack == 'File' and video_args.soundtrack_path is not None:
        if video_args.soundtrack_path.startswith(('http://', 'https://')):
            print(f"Pre-downloading soundtrack at the beginning of the render process: {video_args.soundtrack_path}")
            try:
                from .video_audio_utilities import download_audio
                video_args.soundtrack_path = download_audio(video_args.soundtrack_path)
                print(f"Audio successfully pre-downloaded to: {video_args.soundtrack_path}")
            except Exception as e:
                print(f"Error pre-downloading audio: {e}")

    # Always use experimental render core (legacy core removed)
    experimental_core.render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root)

