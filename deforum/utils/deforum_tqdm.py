import os
from math import ceil
import tqdm
from modules.shared import progress_print_out, opts, cmd_opts
import sys

# Attempt to import ParseqAdapter from the correct location
try:
    from ..integrations.parseq_adapter import ParseqAdapter
except ImportError:
    class ParseqAdapter: # Minimal fallback if not found
        def __init__(self, *args, **kwargs):
            self.use_parseq = False # Ensure use_parseq attribute exists
            pass 

class DeforumTQDM:
    def __init__(self, args, anim_args, parseq_args, video_args, controlnet_args, loop_args):
        self._tqdm = None
        self._args = args
        self._anim_args = anim_args
        self._parseq_args = parseq_args
        self._video_args = video_args
        self._controlnet_args = controlnet_args # Store controlnet_args
        self._loop_args = loop_args # Store loop_args
        
        self.parseq_adapter = None
        try:
            # Pass all required arguments to ParseqAdapter
            self.parseq_adapter = ParseqAdapter(self._parseq_args, self._anim_args, self._video_args, self._controlnet_args, self._loop_args)
        except Exception as e: # Catch any exception during ParseqAdapter instantiation
            print(f"[WARN] Failed to instantiate ParseqAdapter in DeforumTQDM: {e}")
            # Ensure a fallback ParseqAdapter with use_parseq=False if instantiation fails
            if not isinstance(self.parseq_adapter, ParseqAdapter) or not hasattr(self.parseq_adapter, 'use_parseq'):
                 self.parseq_adapter = ParseqAdapter.__new__(ParseqAdapter) # Create instance without calling __init__
                 self.parseq_adapter.use_parseq = False # Manually set use_parseq for fallback

    def reset(self):
        from ..core.keyframe_animation import DeformAnimKeys
        deforum_total = 0
        # FIXME: get only amount of steps
        parseq_adapter = self.parseq_adapter
        keys = DeformAnimKeys(self._anim_args) if not parseq_adapter.use_parseq else parseq_adapter.anim_keys

        start_frame = 0
        if self._anim_args.resume_from_timestring:
            for tmp in os.listdir(self._args.outdir):
                filename = tmp.split("_")
                # don't use saved depth maps to count number of frames
                if self._anim_args.resume_timestring in filename and "depth" not in filename:
                    start_frame += 1
            start_frame = start_frame - 1
        using_vid_init = self._anim_args.animation_mode == 'Video Input'
        turbo_steps = 1 if using_vid_init else int(self._anim_args.diffusion_cadence)
        if self._anim_args.resume_from_timestring:
            last_frame = start_frame - 1
            if turbo_steps > 1:
                last_frame -= last_frame % turbo_steps
            if turbo_steps > 1:
                turbo_next_frame_idx = last_frame
                turbo_prev_frame_idx = turbo_next_frame_idx
                start_frame = last_frame + turbo_steps
        frame_idx = start_frame
        had_first = False
        while frame_idx < self._anim_args.max_frames:
            strength = keys.strength_schedule_series[frame_idx]
            if not had_first and self._args.use_init and ((self._args.init_image is not None and self._args.init_image != '') or self._args.init_image_box is not None):
                deforum_total += int(ceil(self._args.steps * (1 - strength)))
                had_first = True
            elif not had_first:
                deforum_total += self._args.steps
                had_first = True
            else:
                deforum_total += int(ceil(self._args.steps * (1 - strength)))

            if turbo_steps > 1:
                frame_idx += turbo_steps
            else:
                frame_idx += 1

        self._tqdm = tqdm.tqdm(
            desc="Deforum progress",
            total=deforum_total,
            position=1,
            file=progress_print_out
        )

    def update(self):
        if not opts.multiple_tqdm or cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.update()

    def updateTotal(self, new_total):
        if not opts.multiple_tqdm or cmd_opts.disable_console_progressbars:
            return
        if self._tqdm is None:
            self.reset()
        self._tqdm.total = new_total

    def clear(self):
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None

# Custom sys.stdout for TQDM
class TqdmToLogger(object):
    def __init__(self, logger, level=None):
        self.logger = logger
        self.level = level or logging.INFO
        self.buf = ''

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        if self.buf:
            self.logger.log(self.level, self.buf)
            self.buf = '' # Clear buffer after logging

# Optional: Setup logging if needed for TqdmToLogger
import logging
# Example: logging.basicConfig(level=logging.INFO)
# deforum_logger = logging.getLogger('deforum') 
# tqdm_out = TqdmToLogger(deforum_logger, level=logging.INFO)

# Fallback to default tqdm if something goes wrong with DeforumTQDM for stability
# This allows the program to continue running even if custom TQDM has issues.
# However, the original error was ModuleNotFound, so fixing the import path is the primary goal.
_original_tqdm = tqdm.tqdm

def get_tqdm(*args, **kwargs):
    # Check if this is being called with Deforum-specific arguments
    # If args length is 6 and matches our expected Deforum arguments structure, use DeforumTQDM
    if len(args) == 6:
        try:
            return DeforumTQDM(*args, **kwargs)
        except Exception as e:
            print(f"[WARN] DeforumTQDM failed with Deforum args: {e}")
            return _original_tqdm(**kwargs)
    else:
        # This is a standard tqdm call (like from WebUI), use original tqdm
        return _original_tqdm(*args, **kwargs)

tqdm.tqdm = get_tqdm
