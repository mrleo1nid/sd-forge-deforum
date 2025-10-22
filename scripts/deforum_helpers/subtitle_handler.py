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

"""Subtitle handling - Mixed pure and impure functions.

This module contains:
- Pure formatting/calculation functions imported from deforum.utils.subtitle_utils
- Impure I/O functions for writing subtitle files (kept here due to side effects)
"""

from decimal import Decimal

# Import pure functions from refactored utils module
from deforum.utils.subtitle_utils import (
    time_to_srt_format,
    calculate_frame_duration,
    frame_time,
    format_subtitle_value as _format_value,
    SUBTITLE_PARAM_NAMES as param_dict,
    get_user_param_names,
)

# Backward compatibility alias
get_user_values = get_user_param_names


def init_srt_file(filename, fps, precision=20):
    with open(filename, "w") as f:
        pass
    return calculate_frame_duration(fps)


def write_frame_subtitle(filename, frame_number, frame_duration, text):
    # Used by stable core only. Meant to be used when subtitles are intended to change with every frame.
    frames_per_subtitle = 1
    # For low FPS animations and for debugging this is fine, but at higher FPS the file may be too bloated and not fit
    # for YouTube upload, in which case "write_subtitle_from_to" may be used directly with a longer duration instead.
    start_time = frame_time(frame_number, frame_duration)
    end_time = (Decimal(frame_number) + Decimal(frames_per_subtitle)) * frame_duration
    write_subtitle_from_to(filename, frame_number, start_time, end_time, text)


def write_subtitle_from_to(filename, frame_number, start_time_s, end_time_s, text):
    # start_time should be the same as end_time of the last call
    with open(filename, "a", encoding="utf-8") as f:
        # see https://en.wikipedia.org/wiki/SubRip#Format
        f.write(f"{frame_number + 1}\n")
        f.write(f"{time_to_srt_format(start_time_s)} --> {time_to_srt_format(end_time_s)}\n")
        f.write(f"{text}\n\n")


def format_animation_params(keys, prompt_series, frame_idx, params_to_print):
    params_string = ""
    for key, value in param_dict.items():
        if value['user'] in params_to_print:
            backend_key = value['backend']
            print_key = value['print']
            param_value = getattr(keys, backend_key)[frame_idx]
            formatted_value = _format_value(param_value)
            params_string += f"{print_key}: {formatted_value}; "

    if "Prompt" in params_to_print:
        params_string += f"Prompt: {prompt_series[frame_idx]}; "

    params_string = params_string.rstrip("; ")  # Remove trailing semicolon and whitespace
    return params_string
