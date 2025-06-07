"""
Animation Key Frame Processing

Contains classes for handling animation key frames and scheduling.
Note: FreeU and Kohya HR Fix functionality has been removed.
"""

import re
try:
    import numpy as np
    import pandas as pd
    import numexpr
    PANDAS_AVAILABLE = True
except ImportError:
    # Create minimal stubs for when pandas is not available
    PANDAS_AVAILABLE = False
    np = None
    pd = None
    numexpr = None
from ..prompt.core_prompt_processing import check_is_number
from modules import scripts, shared

class DeformAnimKeys():
    def __init__(self, anim_args, seed=-1):
        self.fi = FrameInterpolater(anim_args.max_frames, seed)
        self.angle_series = self.fi.parse_inbetweens(anim_args.angle, 'angle')
        self.zoom_series = self.fi.parse_inbetweens(anim_args.zoom, 'zoom')
        self.translation_x_series = self.fi.parse_inbetweens(anim_args.translation_x, 'translation_x')
        self.translation_y_series = self.fi.parse_inbetweens(anim_args.translation_y, 'translation_y')
        self.translation_z_series = self.fi.parse_inbetweens(anim_args.translation_z, 'translation_z')
        self.rotation_3d_x_series = self.fi.parse_inbetweens(anim_args.rotation_3d_x, 'rotation_3d_x')
        self.rotation_3d_y_series = self.fi.parse_inbetweens(anim_args.rotation_3d_y, 'rotation_3d_y')
        self.rotation_3d_z_series = self.fi.parse_inbetweens(anim_args.rotation_3d_z, 'rotation_3d_z')
        self.perspective_flip_theta_series = self.fi.parse_inbetweens(anim_args.perspective_flip_theta, 'perspective_flip_theta')
        self.perspective_flip_phi_series = self.fi.parse_inbetweens(anim_args.perspective_flip_phi, 'perspective_flip_phi')
        self.perspective_flip_gamma_series = self.fi.parse_inbetweens(anim_args.perspective_flip_gamma, 'perspective_flip_gamma')
        self.perspective_flip_fv_series = self.fi.parse_inbetweens(anim_args.perspective_flip_fv, 'perspective_flip_fv')
        self.transform_center_x_series = self.fi.parse_inbetweens(anim_args.transform_center_x, 'transform_center_x')
        self.transform_center_y_series = self.fi.parse_inbetweens(anim_args.transform_center_y, 'transform_center_y')
        self.noise_schedule_series = self.fi.parse_inbetweens(anim_args.noise_schedule, 'noise_schedule')
        self.strength_schedule_series = self.fi.parse_inbetweens(anim_args.strength_schedule, 'strength_schedule')
        self.keyframe_strength_schedule_series = self.fi.parse_inbetweens(anim_args.keyframe_strength_schedule, 'keyframe_strength_schedule')
        self.contrast_schedule_series = self.fi.parse_inbetweens(anim_args.contrast_schedule, 'contrast_schedule')
        self.cfg_scale_schedule_series = self.fi.parse_inbetweens(anim_args.cfg_scale_schedule, 'cfg_scale_schedule')
        self.distilled_cfg_scale_schedule_series = self.fi.parse_inbetweens(anim_args.distilled_cfg_scale_schedule, 'distilled_cfg_scale_schedule')
        self.fov_series = self.fi.parse_inbetweens(anim_args.fov_schedule, 'fov_schedule')
        self.aspect_ratio_series = self.fi.parse_inbetweens(anim_args.aspect_ratio_schedule, 'aspect_ratio_schedule')
        self.near_series = self.fi.parse_inbetweens(anim_args.near_schedule, 'near_schedule')
        self.far_series = self.fi.parse_inbetweens(anim_args.far_schedule, 'far_schedule')
        self.seed_schedule_series = self.fi.parse_inbetweens(anim_args.seed_schedule, 'seed_schedule')
        self.mask_schedule_series = self.fi.parse_inbetweens(anim_args.mask_schedule, 'mask_schedule', is_single_string=True)
        self.noise_mask_schedule_series = self.fi.parse_inbetweens(anim_args.noise_mask_schedule, 'noise_mask_schedule', is_single_string=True)
        self.kernel_schedule_series = self.fi.parse_inbetweens(anim_args.kernel_schedule, 'kernel_schedule')
        self.sigma_schedule_series = self.fi.parse_inbetweens(anim_args.sigma_schedule, 'sigma_schedule')
        self.amount_schedule_series = self.fi.parse_inbetweens(anim_args.amount_schedule, 'amount_schedule')
        self.threshold_schedule_series = self.fi.parse_inbetweens(anim_args.threshold_schedule, 'threshold_schedule')
        self.cadence_flow_factor_schedule_series = self.fi.parse_inbetweens(anim_args.cadence_flow_factor_schedule, 'cadence_flow_factor_schedule')
        self.redo_flow_factor_schedule_series = self.fi.parse_inbetweens(anim_args.redo_flow_factor_schedule, 'redo_flow_factor_schedule')
        
        # Only enable scheduling features that don't depend on obsolete extensions
        if anim_args.enable_steps_scheduling:
            self.steps_schedule_series = self.fi.parse_inbetweens(anim_args.steps_schedule, 'steps_schedule')
        else:
            self.steps_schedule_series = None
        if anim_args.enable_sampler_scheduling:
            self.sampler_schedule_series = self.fi.parse_inbetweens(anim_args.sampler_schedule, 'sampler_schedule', is_single_string=True)
        else:
            self.sampler_schedule_series = None
        if anim_args.enable_scheduler_scheduling:
            self.scheduler_schedule_series = self.fi.parse_inbetweens(anim_args.scheduler_schedule, 'scheduler_schedule', is_single_string=True)
        else:
            self.scheduler_schedule_series = None
        if anim_args.enable_checkpoint_scheduling:
            self.checkpoint_schedule_series = self.fi.parse_inbetweens(anim_args.checkpoint_schedule, 'checkpoint_schedule', is_single_string=True)
        else:
            self.checkpoint_schedule_series = None
        if anim_args.enable_clipskip_scheduling:
            self.clipskip_schedule_series = self.fi.parse_inbetweens(anim_args.clipskip_schedule, 'clipskip_schedule')
        else:
            self.clipskip_schedule_series = None
        if anim_args.enable_noise_multiplier_scheduling:
            self.noise_multiplier_schedule_series = self.fi.parse_inbetweens(anim_args.noise_multiplier_schedule, 'noise_multiplier_schedule')
        else:
            self.noise_multiplier_schedule_series = None
        if anim_args.enable_ddim_eta_scheduling:
            self.ddim_eta_schedule_series = self.fi.parse_inbetweens(anim_args.ddim_eta_schedule, 'ddim_eta_schedule')
        else:
            self.ddim_eta_schedule_series = None
        if anim_args.enable_ancestral_eta_scheduling:
            self.ancestral_eta_schedule_series = self.fi.parse_inbetweens(anim_args.ancestral_eta_schedule, 'ancestral_eta_schedule')
        else:
            self.ancestral_eta_schedule_series = None
        if anim_args.enable_subseed_scheduling:
            self.subseed_schedule_series = self.fi.parse_inbetweens(anim_args.subseed_schedule, 'subseed_schedule')
            self.subseed_strength_schedule_series = self.fi.parse_inbetweens(anim_args.subseed_strength_schedule, 'subseed_strength_schedule')
        else:
            self.subseed_schedule_series = None
            self.subseed_strength_schedule_series = None

class ControlNetKeys():
    def __init__(self, anim_args, controlnet_args):
        self.fi = FrameInterpolater(anim_args.max_frames)
        cn_max_models_num = 5
        for cn_model_index in range(1, cn_max_models_num+1):
            for suffix in ['weight', 'guidance_start', 'guidance_end']:
                prefix = f"cn_{cn_model_index}"
                input_key = f"{prefix}_{suffix}"
                output_key = f"{input_key}_schedule_series"
                try:
                    input_value = getattr(controlnet_args, input_key)
                    schedule_data = self.fi.parse_inbetweens(input_value, input_key)
                    setattr(self, output_key, schedule_data)
                except Exception:
                    setattr(self, output_key, [1.0] * anim_args.max_frames)

class LooperAnimKeys():
    def __init__(self, loop_args, anim_args, seed):
        self.fi = FrameInterpolater(anim_args.max_frames, seed)
        self.use_looper = loop_args.use_looper
        self.imagesToKeyframe = loop_args.init_images
        self.image_strength_schedule_series = self.fi.parse_inbetweens(loop_args.image_strength_schedule, 'image_strength_schedule')
        self.image_keyframe_strength_schedule_series = self.fi.parse_inbetweens(loop_args.image_keyframe_strength_schedule, 'image_keyframe_strength_schedule')
        self.blendFactorMax_series = self.fi.parse_inbetweens(loop_args.blendFactorMax, 'blendFactorMax')
        self.blendFactorSlope_series = self.fi.parse_inbetweens(loop_args.blendFactorSlope, 'blendFactorSlope')
        self.tweening_frames_schedule_series = self.fi.parse_inbetweens(loop_args.tweening_frames_schedule, 'tweening_frames_schedule')
        self.color_correction_factor_series = self.fi.parse_inbetweens(loop_args.color_correction_factor, 'color_correction_factor')

class FrameInterpolater():
    def __init__(self, max_frames=0, seed=-1) -> None:
        self.max_frames = max_frames
        self.seed = seed

    def parse_inbetweens(self, value, filename = 'unknown', is_single_string = False):
        is_integer_schedule = (filename == 'seed_schedule') # Add other integer schedules if any
        # Add other integer schedules like steps_schedule, clipskip_schedule if they need to be integers
        if filename in ['steps_schedule', 'clipskip_schedule', 'kernel_schedule']:
            is_integer_schedule = True
        return self.get_inbetweens(self.parse_key_frames(value, filename), integer=is_integer_schedule, is_single_string=is_single_string, filename=filename)

    def sanitize_value(self, value):
        return value.replace("'", "").replace('"', "").replace('(', "").replace(')', "")

    def get_inbetweens(self, key_frames, integer=False, interp_method='Linear', is_single_string = False, filename = 'unknown'):
        if not PANDAS_AVAILABLE:
            # Fallback: return simple list with default values
            print(f"Warning: pandas not available, using basic fallback for {filename}")
            if is_single_string:
                return [str(list(key_frames.values())[0] if key_frames else "")] * self.max_frames
            else:
                return [float(list(key_frames.values())[0] if key_frames else 0.0)] * self.max_frames
                
        key_frame_series = pd.Series([np.nan for _ in range(self.max_frames)], dtype=object)

        for i, key_str_val in key_frames.items():
            if is_single_string:
                key_frame_series[i] = self.sanitize_value(str(key_str_val))
            elif filename == 'seed_schedule':
                if isinstance(key_str_val, str):
                    try:
                        key_frame_series[i] = float(key_str_val)
                    except ValueError:
                        key_frame_series[i] = str(key_str_val).lower() 
                else:
                    key_frame_series[i] = float(key_str_val)
            else:
                try:
                    key_frame_series[i] = float(key_str_val)
                except ValueError:
                    print(f"Warning: Could not convert value '{key_str_val}' to float for schedule '{filename}' at frame {i}. Using NaN.")
                    key_frame_series[i] = np.nan

        if is_single_string:
            key_frame_series = key_frame_series.ffill().bfill()
        else:
            if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
                interp_method = 'Linear'
            
            numeric_series_for_interp = pd.to_numeric(key_frame_series, errors='coerce')
            interpolated_numeric_series = numeric_series_for_interp.interpolate(method=interp_method.lower(), limit_direction='both')
            
            output_series = pd.Series([np.nan for _ in range(self.max_frames)], dtype=object)
            for i in range(self.max_frames):
                if isinstance(key_frame_series[i], str) and not pd.isna(key_frame_series[i]):
                    output_series[i] = key_frame_series[i]
                else:
                    output_series[i] = interpolated_numeric_series[i]
            key_frame_series = output_series.ffill().bfill()

        if integer:
            if filename == 'seed_schedule':
                def convert_seed_value(val):
                    if isinstance(val, str) and val.lower() == 's':
                        return self.seed
                    try:
                        return int(float(val))
                    except (ValueError, TypeError):
                        if pd.isna(val):
                            return self.seed 
                        return self.seed
                key_frame_series = key_frame_series.apply(convert_seed_value).astype(int)
            else:
                key_frame_series = pd.to_numeric(key_frame_series, errors='raise').astype(int)

        return key_frame_series

    def parse_key_frames(self, string, filename='unknown'):
        # Handle None or non-string values
        if string is None:
            print(f"Warning: {filename} schedule is None, using default '0:(0)'")
            string = "0:(0)"
        elif not isinstance(string, str):
            print(f"Warning: {filename} schedule is not a string, converting: {string}")
            string = str(string)
            
        # because math functions (i.e. sin(t)) can utilize brackets 
        # it extracts the value in form of some stuff
        # which has previously been enclosed with brackets and
        # with a comma or end of line existing after the closing one
        import re
        pattern = r'((?P<frame>[0-9]+):[\s]*\((?P<param>[\S\s]*?)\))'
        frames = dict()
        for match_object in re.finditer(pattern, string):
            frame = int(match_object.groupdict()['frame'])
            param = match_object.groupdict()['param']
            if frame in frames:
                try:
                    frames[frame] = float(param)
                except:
                    frames[frame] = param
            else:
                try:
                    frames[frame] = float(param)
                except:
                    frames[frame] = param

        if frames == {} and len(string) != 0:
            raise RuntimeError('Key Frame string not correctly formatted')
        return frames