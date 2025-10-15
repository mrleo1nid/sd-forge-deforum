"""
Argument Transformations - Fixed for Missing Components
Contains argument processing, transformations, and component management logic
"""

import dataclasses
import json
import os
import time
from types import SimpleNamespace
import pathlib
import traceback
from .general_utils import substitute_placeholders, get_deforum_version, clean_gradio_path_strings
from ..utils import log_utils
from ..utils.color_constants import BOLD, CYAN, RESET_COLOR

# Conditional imports
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from modules.processing import get_fixed_seed
    WEBUI_AVAILABLE = True
except ImportError:
    WEBUI_AVAILABLE = False
    
    def get_fixed_seed(seed):
        """Fallback seed function."""
        if seed == -1:
            import random
            return random.randint(0, 2**32 - 1)
        return seed

try:
    from .arg_defaults import RootArgs, DeforumAnimArgs, DeforumArgs, ParseqArgs, WanArgs, DeforumOutputArgs
    from .arg_validation import sanitize_strength, sanitize_seed
    ARG_MODULES_AVAILABLE = True
except ImportError:
    ARG_MODULES_AVAILABLE = False

try:
    from ..integrations.controlnet.core_integration import controlnet_component_names
    CONTROLNET_AVAILABLE = True
except ImportError:
    CONTROLNET_AVAILABLE = False
    
    def controlnet_component_names():
        """Fallback controlnet function."""
        return []

try:
    from .general_utils import substitute_placeholders
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    
    def substitute_placeholders(text, arg_list, base_path):
        """Fallback substitute function."""
        return text

try:
    from ..models.data_models import (
        AnimationArgs, DeforumArgs as DeforumArgs_Dataclass, # Aliasing to avoid conflict
        DeforumOutputArgs as DeforumOutputArgs_Dataclass, 
        ParseqArgs as ParseqArgs_Dataclass, RootArgs as RootArgs_Dataclass, 
        WanArgs as WanArgs_Dataclass, LoopArgs as LoopArgs_Dataclass, 
        ControlnetArgs as ControlnetArgs_Dataclass,
        create_animation_args_from_dict, 
        create_deforum_args_from_dict, 
        create_deforum_output_args_from_dict, 
        create_parseq_args_from_dict, create_wan_args_from_dict, create_root_args_from_dict, 
        create_loop_args_from_dict, create_controlnet_args_from_dict
    )
    # ADD IMPORT FOR THE CORRECT ANIMATIONARGS DATACLASS
    from .argument_models import DeforumAnimationArgs, DeforumGenerationArgs, DeforumVideoArgs, ParseqArgs as ParseqArgs_Config, WanArgs as WanArgs_Config # Use DeforumVideoArgs for video_args
    DATACLASSES_AVAILABLE = True
except ImportError as e:
    print(f"[DEBUG] Failed to import dataclasses in arg_transformations: {e}")
    DATACLASSES_AVAILABLE = False
    # Define fallback placeholders if dataclasses are not available
    class PlaceholderArgs: pass
    AnimationArgs, DeforumArgs_Dataclass, DeforumOutputArgs_Dataclass, ParseqArgs_Dataclass, RootArgs_Dataclass, WanArgs_Dataclass, LoopArgs_Dataclass, ControlnetArgs_Dataclass = (PlaceholderArgs,) * 8 # VideoArgs removed
    def create_animation_args_from_dict(d): return SimpleNamespace(**d)
    def create_deforum_args_from_dict(d): return SimpleNamespace(**d)
    def create_deforum_output_args_from_dict(d): return SimpleNamespace(**d) # Added
    def create_parseq_args_from_dict(d): return SimpleNamespace(**d)
    def create_wan_args_from_dict(d): return SimpleNamespace(**d)
    def create_root_args_from_dict(d): return SimpleNamespace(**d)
    def create_loop_args_from_dict(d): return SimpleNamespace(**d)
    def create_controlnet_args_from_dict(d): return SimpleNamespace(**d) # Added


def get_component_names():
    """Get all component names for UI binding - CORRECTED ORDER VERSION."""
    # Define essential components in the EXACT order they appear in the UI
    # This order MUST match how components are created in the UI modules
    essential_components = [
        # Batch mode components (first in UI)
        'override_settings_with_file', 
        'custom_settings_file',
        
        # Basic generation components (Run tab order)
        'seed', 'steps', 'sampler', 'scheduler', 'checkpoint', 'clip_skip',
        'W', 'H', 'strength', 'cfg_scale', 'distilled_cfg_scale', 'tiling',
        'restore_faces', 'seed_resize_from_w', 'seed_resize_from_h',
        'noise_multiplier', 'ddim_eta', 'ancestral_eta', 
        
        # Init/Mask components (Init tab order)
        'use_init', 'use_mask', 'invert_mask', 'overlay_mask',
        'mask_file', 'mask_overlay_blur', 'mask_brightness_adjust', 'mask_contrast_adjust',
        'fill', 'full_res_mask', 'full_res_mask_padding',
        'init_image', 'init_image_box',
        
        # Prompt and behavior components
        'prompt', 'negative_prompt', 'animation_prompts', 
        'animation_prompts_positive', 'animation_prompts_negative',
        'seed_behavior', 'seed_iter_N', 'subseed', 'subseed_strength',
        
        # Animation components (Animation tab order)
        'animation_mode', 'max_frames', 'border', 
        'angle', 'zoom', 'translation_x', 'translation_y', 'translation_z', 
        'rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z',
        'flip_2d_perspective', 'perspective_flip_theta', 'perspective_flip_phi',
        'perspective_flip_gamma', 'perspective_flip_fv', 'enable_perspective_flip',
        
        # Schedule components (Keyframes tab order)
        'noise_schedule', 'strength_schedule', 'contrast_schedule', 
        'cfg_scale_schedule', 'distilled_cfg_scale_schedule', 'steps_schedule', 'seed_schedule',
        
        # Advanced animation components
        'optical_flow_cadence', 'cadence_flow_factor_schedule', 
        'optical_flow_redo_generation', 'redo_flow_factor_schedule',
        'diffusion_redo', 'motion_preview_mode', 'motion_preview_length', 'motion_preview_step',
        
        # Noise and effects
        'noise_type', 'perlin_octaves', 'perlin_persistence',
        'color_force_grayscale', 'color_coherence', 
        'color_coherence_video_every_N_frames', 'color_coherence_image_path',
        
        # Depth and 3D
        'depth_algorithm', 'midas_weight', 'depth_warp_msg_html',
        
        # Output and processing
        'save_settings', 'save_sample', 'display_samples', 'save_sample_per_step',
        'show_sample_per_step', 'override_these_with_webui', 'batch_name',
        'filename_format', 'use_areas', 'reroll_blank_frames',
        
        # Video output components
        'skip_video_creation', 'make_gif', 
        'frame_interp_slow_mo_amount', 'frame_interp_x_amount',
        'ncnn_upscale_model', 'ncnn_upscale_factor',
        'aspect_ratio_use_old_formula', 'aspect_ratio_schedule'
    ]
    
    # Add dynamic components if available
    try:
        if ARG_MODULES_AVAILABLE:
            essential_components.extend(DeforumAnimArgs().keys())
            essential_components.extend(DeforumArgs().keys()) 
            essential_components.extend(DeforumOutputArgs().keys())
            essential_components.extend(ParseqArgs().keys())
            essential_components.extend(WanArgs().keys())
    except:
        pass
    
    # Add ControlNet components if available
    try:
        if CONTROLNET_AVAILABLE:
            essential_components.extend(controlnet_component_names())
    except:
        pass
    
    # Remove duplicates while preserving order
    seen = set()
    unique_components = []
    for item in essential_components:
        if item not in seen:
            seen.add(item)
            unique_components.append(item)
    
    # Debug component count and critical positions
    critical_components = ['W', 'H', 'strength', 'animation_prompts', 'mask_overlay_blur']
    print(f"🔧 Component names list: {len(unique_components)} total components")
    for comp in critical_components:
        if comp in unique_components:
            pos = unique_components.index(comp)
            print(f"   {comp:20} at position {pos:3d}")
        else:
            print(f"   {comp:20} -> MISSING!")
    
    return unique_components


def get_settings_component_names():
    """Get settings component names."""
    return [name for name in get_component_names()]


def pack_args(args_dict, keys_function):
    """Pack arguments using specified keys function or dataclass - with fallback for missing keys."""
    result = {}
    
    # Handle both old callable style and new dataclass style
    if hasattr(keys_function, '__dataclass_fields__'):
        # It's a dataclass, get field names
        keys = list(keys_function.__dataclass_fields__.keys())
    elif callable(keys_function):
        # It's a callable, call it to get keys
        keys = keys_function()
    else:
        # Fallback: try to iterate directly
        keys = list(keys_function)
    
    for name in keys:
        if name in args_dict:
            result[name] = args_dict[name]
        else:
            # Provide safe defaults for missing components
            if name in ['override_settings_with_file']:
                result[name] = False
            elif name in ['custom_settings_file']:
                result[name] = None
            else:
                result[name] = None
    return result


def create_namespace_from_dict(args_dict: dict, keys_function) -> SimpleNamespace:
    """Create SimpleNamespace from dictionary using keys function.
    
    Args:
        args_dict: Source dictionary
        keys_function: Function returning valid keys
        
    Returns:
        SimpleNamespace object with filtered arguments
    """
    filtered_args = pack_args(args_dict, keys_function)
    return SimpleNamespace(**filtered_args)


def process_animation_prompts(args_dict_main: dict, root: SimpleNamespace) -> None:
    """Process and transform animation prompts.
    
    Args:
        args_dict_main: Main arguments dictionary
        root: Root namespace to update
    """
    # Parse animation prompts JSON - with fallback
    try:
        animation_prompts_raw = args_dict_main.get('animation_prompts', '{"0": "a beautiful landscape"}')
        if isinstance(animation_prompts_raw, str):
            root.animation_prompts = json.loads(animation_prompts_raw)
        else:
            root.animation_prompts = animation_prompts_raw if animation_prompts_raw else {"0": "a beautiful landscape"}
    except (json.JSONDecodeError, TypeError):
        print("⚠️ Warning: Invalid animation prompts JSON, using default")
        root.animation_prompts = {"0": "a beautiful landscape"}
    
    # Get positive and negative prompts - with fallbacks
    positive_prompts = args_dict_main.get('animation_prompts_positive', '')
    negative_prompts = args_dict_main.get('animation_prompts_negative', '')
    
    # Clean negative prompts
    if negative_prompts:
        negative_prompts = negative_prompts.replace('--neg', '')
    
    # Create prompt keyframes
    root.prompt_keyframes = [key for key in root.animation_prompts.keys()]
    
    # Combine prompts with proper negative prompt formatting
    if positive_prompts or negative_prompts:
        root.animation_prompts = {
            key: f"{positive_prompts} {val} {'' if '--neg' in val else '--neg'} {negative_prompts}"
            for key, val in root.animation_prompts.items()
        }


def process_seed_settings(args: SimpleNamespace, root: SimpleNamespace) -> None:
    """Process seed settings and store raw seed.
    
    Args:
        args: Arguments namespace
        root: Root namespace
    """
    # Store raw seed before processing
    if hasattr(args, 'seed') and args.seed == -1:
        root.raw_seed = -1
    else:
        root.raw_seed = getattr(args, 'seed', -1)
    
    # Get fixed seed
    if hasattr(args, 'seed'):
        args.seed = get_fixed_seed(args.seed)
    
    # Update raw seed if it wasn't random
    if root.raw_seed != -1:
        root.raw_seed = args.seed


def setup_output_directory(args: SimpleNamespace, root: SimpleNamespace, p, current_arg_list: list) -> None:
    """Setup output directory and batch name processing.
    
    Args:
        args: Arguments namespace
        root: Root namespace
        p: Processing object
        current_arg_list: List of argument objects for substitution
    """
    # Store raw batch name
    root.raw_batch_name = getattr(args, 'batch_name', 'Deforum')
    
    # Process batch name with substitutions
    full_base_folder_path = os.path.join(os.getcwd(), p.outpath_samples)
    args.batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)
    
    # Setup output directory
    args.outdir = os.path.join(p.outpath_samples, str(args.batch_name))
    args.outdir = os.path.join(os.getcwd(), args.outdir)
    args.outdir = os.path.realpath(args.outdir)
    
    # Create directory
    os.makedirs(args.outdir, exist_ok=True)


def setup_default_image(args: SimpleNamespace, root: SimpleNamespace) -> None:
    """Setup default image for processing.
    
    Args:
        args: Arguments namespace
        root: Root namespace
    """
    if not PIL_AVAILABLE:
        return
        
    try:
        default_img_path = os.path.join(pathlib.Path(__file__).parent.absolute(), '114763196.jpg')
        if os.path.exists(default_img_path):
            default_img = Image.open(default_img_path)
            assert default_img is not None
            W = getattr(args, 'W', 512)
            H = getattr(args, 'H', 512)
            default_img = default_img.resize((W, H))
            root.default_img = default_img
    except Exception as e:
        print(f"⚠️ Warning: Could not load default image: {e}")


def process_init_image_settings(args: SimpleNamespace, anim_args: SimpleNamespace) -> None:
    """Process initial image settings.
    
    Args:
        args: Arguments namespace
        anim_args: Animation arguments namespace
    """
    use_init = getattr(args, 'use_init', False)
    hybrid_use_init_image = getattr(anim_args, 'hybrid_use_init_image', False)
    
    if not use_init and not hybrid_use_init_image:
        args.init_image = None
        args.init_image_box = None


def create_additional_substitutions() -> SimpleNamespace:
    """Create additional substitution variables.
    
    Returns:
        SimpleNamespace with date and time substitutions
    """
    return SimpleNamespace(
        date=time.strftime('%Y%m%d'),
        time=time.strftime('%H%M%S')
    )


def process_args(args_dict, index=0):
    """
    Process and transform arguments for Deforum execution - FIXED VERSION.
    Note: FreeU and Kohya HR Fix functionality has been removed.
    
    Returns:
        tuple: (args_loaded_ok, root, args, anim_args, video_args, parseq_args, loop_args, controlnet_args, wan_args)
    """
    try:
        if not DATACLASSES_AVAILABLE:
            log_utils.error("Dataclasses not available. Falling back to SimpleNamespace. THIS IS UNEXPECTED.")
            # Fallback to old behavior if dataclasses cannot be imported
            _args = SimpleNamespace(**args_dict.get('args', {})) # Renamed to avoid conflict with outer scope 'args'
            _anim_args = SimpleNamespace(**args_dict.get('anim_args', {})) # Renamed
            _video_args_ns = SimpleNamespace(**args_dict.get('video_args', {})) # Renamed
            _parseq_args = SimpleNamespace(**args_dict.get('parseq_args', {})) # Renamed
            _loop_args = SimpleNamespace(**args_dict.get('loop_args', {})) # Renamed
            _controlnet_args_ns = SimpleNamespace(**args_dict.get('controlnet_args', {})) # Renamed
            _wan_args = SimpleNamespace(**args_dict.get('wan_args', {})) # Renamed
            _root = SimpleNamespace() # Renamed
            return True, _root, _args, _anim_args, _video_args_ns, _parseq_args, _loop_args, _controlnet_args_ns, _wan_args

        # DeforumGenerationArgs (aliased as 'args' in original return tuple)
        gen_arg_fields = set(DeforumGenerationArgs.__dataclass_fields__.keys())
        dict_for_gen_args = {k: args_dict[k] for k in gen_arg_fields if k in args_dict and args_dict[k] is not None}
        
        # Fix: Validate critical generation args to prevent component mapping errors
        if 'strength' in dict_for_gen_args:
            if not isinstance(dict_for_gen_args['strength'], (int, float)) or dict_for_gen_args['strength'] < 0.0 or dict_for_gen_args['strength'] > 1.0:
                print(f"⚠️ Invalid strength value {dict_for_gen_args['strength']}, using default 0.85")
                dict_for_gen_args['strength'] = 0.85
                
        if 'cfg_scale' in dict_for_gen_args:
            if not isinstance(dict_for_gen_args['cfg_scale'], (int, float)) or dict_for_gen_args['cfg_scale'] < 1.0 or dict_for_gen_args['cfg_scale'] > 30.0:
                print(f"⚠️ Invalid cfg_scale value {dict_for_gen_args['cfg_scale']}, using default 7.0")
                dict_for_gen_args['cfg_scale'] = 7.0
        
        # TODO: Handle Enum conversions for DeforumGenerationArgs if any field needs it
        args = DeforumGenerationArgs(**dict_for_gen_args) # Instantiated 'args'

        # DeforumAnimationArgs (aliased as 'anim_args')
        anim_arg_fields = set(DeforumAnimationArgs.__dataclass_fields__.keys())
        dict_for_anim_args = {k: args_dict[k] for k in anim_arg_fields if k in args_dict}
        
        # Fix: Validate critical animation args to prevent component mapping errors
        if 'max_frames' in dict_for_anim_args:
            if not isinstance(dict_for_anim_args['max_frames'], int) or dict_for_anim_args['max_frames'] <= 0:
                print(f"⚠️ Invalid max_frames value '{dict_for_anim_args['max_frames']}' (type: {type(dict_for_anim_args['max_frames'])}), using default 334")
                dict_for_anim_args['max_frames'] = 334
        
        # Add validation for other common mismatched fields
        if 'diffusion_cadence' in dict_for_anim_args:
            if not isinstance(dict_for_anim_args['diffusion_cadence'], int) or dict_for_anim_args['diffusion_cadence'] <= 0:
                print(f"⚠️ Invalid diffusion_cadence value '{dict_for_anim_args['diffusion_cadence']}', using default 10")
                dict_for_anim_args['diffusion_cadence'] = 10

        # Fix: Validate perlin_persistence (must be 0.0-1.0 float)
        if 'perlin_persistence' in dict_for_anim_args:
            if not isinstance(dict_for_anim_args['perlin_persistence'], (int, float)) or dict_for_anim_args['perlin_persistence'] < 0.0 or dict_for_anim_args['perlin_persistence'] > 1.0:
                print(f"⚠️ Invalid perlin_persistence value '{dict_for_anim_args['perlin_persistence']}', using default 0.5")
                dict_for_anim_args['perlin_persistence'] = 0.5

        # Fix: Validate perlin_octaves (must be positive int)
        if 'perlin_octaves' in dict_for_anim_args:
            if not isinstance(dict_for_anim_args['perlin_octaves'], int) or dict_for_anim_args['perlin_octaves'] <= 0:
                print(f"⚠️ Invalid perlin_octaves value '{dict_for_anim_args['perlin_octaves']}', using default 4")
                dict_for_anim_args['perlin_octaves'] = 4

        if 'animation_mode' in dict_for_anim_args:
            from ..models.data_models import AnimationMode # Enum from data_models
            if dict_for_anim_args['animation_mode'] is None or dict_for_anim_args['animation_mode'] == 'None':
                # None value or "None" string, use default (silently)
                del dict_for_anim_args['animation_mode'] # Let dataclass default apply
            elif isinstance(dict_for_anim_args['animation_mode'], str):
                try:
                    dict_for_anim_args['animation_mode'] = AnimationMode(dict_for_anim_args['animation_mode'])
                except ValueError:
                    # Only warn for unexpected invalid values, not "None"
                    if dict_for_anim_args['animation_mode'] != 'None':
                        log_utils.warn(f"Invalid string for AnimationMode: {dict_for_anim_args['animation_mode']}. Default will be used.")
                    del dict_for_anim_args['animation_mode'] # Let dataclass default apply

        if 'border' in dict_for_anim_args:
            from ..models.data_models import BorderMode # Enum from data_models
            if dict_for_anim_args['border'] is None:
                # None value, use default
                del dict_for_anim_args['border'] # Let dataclass default apply
            elif isinstance(dict_for_anim_args['border'], str):
                try:
                    dict_for_anim_args['border'] = BorderMode(dict_for_anim_args['border'])
                except ValueError:
                    log_utils.warn(f"Invalid string for BorderMode: {dict_for_anim_args['border']}. Default will be used.")
                    del dict_for_anim_args['border']
        
        # Add other enum conversions for DeforumAnimationArgs as needed (e.g. color_coherence, noise_type)

        anim_args = DeforumAnimationArgs(**dict_for_anim_args) # Instantiated 'anim_args'

        # DeforumVideoArgs (aliased as 'video_args') - was DeforumOutputArgs previously
        video_arg_fields = set(DeforumVideoArgs.__dataclass_fields__.keys())
        dict_for_video_args = {k: args_dict[k] for k in video_arg_fields if k in args_dict and args_dict[k] is not None}
        # TODO: Handle Enum conversions for DeforumVideoArgs if any field needs it (e.g. add_soundtrack if it becomes an enum)
        video_args = DeforumVideoArgs(**dict_for_video_args) # Instantiated 'video_args'
        
        # For ParseqArgs, WanArgs, LoopArgs, ControlnetArgs - currently using create_..._from_dict from data_models
        # This might need similar adjustment if they also have specific dataclasses in argument_models.py
        # For now, keeping them as they were if they don't cause immediate issues.
        raw_parseq_args = {k: args_dict[k] for k in ParseqArgs_Config.__dataclass_fields__ if k in args_dict and args_dict[k] is not None} if DATACLASSES_AVAILABLE and hasattr(ParseqArgs_Config, '__dataclass_fields__') else args_dict.get('parseq_args', {})
        parseq_args = ParseqArgs_Config(**raw_parseq_args) if DATACLASSES_AVAILABLE and hasattr(ParseqArgs_Config, '__dataclass_fields__') else create_parseq_args_from_dict(args_dict.get('parseq_args', {}))
        
        raw_loop_args = {k: args_dict[k] for k in LoopArgs_Dataclass.__dataclass_fields__ if k in args_dict and args_dict[k] is not None} if DATACLASSES_AVAILABLE and hasattr(LoopArgs_Dataclass, '__dataclass_fields__') else args_dict.get('loop_args', {})
        loop_args = LoopArgs_Dataclass(**raw_loop_args) if DATACLASSES_AVAILABLE and hasattr(LoopArgs_Dataclass, '__dataclass_fields__') else create_loop_args_from_dict(args_dict.get('loop_args', {}))

        raw_controlnet_args = {k: args_dict[k] for k in ControlnetArgs_Dataclass.__dataclass_fields__ if k in args_dict and args_dict[k] is not None} if DATACLASSES_AVAILABLE and hasattr(ControlnetArgs_Dataclass, '__dataclass_fields__') else args_dict.get('controlnet_args', {})
        
        # Fix: Validate ControlNet schedule fields to prevent component mapping errors
        if DATACLASSES_AVAILABLE and hasattr(ControlnetArgs_Dataclass, '__dataclass_fields__'):
            schedule_fields = ['cn_1_guidance_start', 'cn_1_guidance_end', 'cn_1_weight', 
                              'cn_2_guidance_start', 'cn_2_guidance_end', 'cn_2_weight',
                              'cn_3_guidance_start', 'cn_3_guidance_end', 'cn_3_weight',
                              'cn_4_guidance_start', 'cn_4_guidance_end', 'cn_4_weight',
                              'cn_5_guidance_start', 'cn_5_guidance_end', 'cn_5_weight']
            
            processor_res_fields = ['cn_1_processor_res', 'cn_2_processor_res', 'cn_3_processor_res', 'cn_4_processor_res', 'cn_5_processor_res']
            threshold_fields = ['cn_1_threshold_a', 'cn_1_threshold_b', 'cn_2_threshold_a', 'cn_2_threshold_b', 
                               'cn_3_threshold_a', 'cn_3_threshold_b', 'cn_4_threshold_a', 'cn_4_threshold_b',
                               'cn_5_threshold_a', 'cn_5_threshold_b']
            
            for field in schedule_fields:
                if field in raw_controlnet_args:
                    val = raw_controlnet_args[field]
                    # Check if it looks like a valid schedule format
                    if not isinstance(val, str) or not ('(' in val and ')' in val and ':' in val):
                        print(f"⚠️ Invalid ControlNet schedule '{field}' = '{val}', using default")
                        if 'start' in field:
                            raw_controlnet_args[field] = "0:(0.0)"
                        elif 'end' in field:
                            raw_controlnet_args[field] = "0:(1.0)"
                        elif 'weight' in field:
                            raw_controlnet_args[field] = "0:(1)"
            
            # Fix processor_res fields
            for field in processor_res_fields:
                if field in raw_controlnet_args:
                    val = raw_controlnet_args[field]
                    if not isinstance(val, int) or val < 64 or val > 2048:
                        print(f"⚠️ Invalid ControlNet processor_res '{field}' = '{val}', using default 64")
                        raw_controlnet_args[field] = 64
            
            # Fix threshold fields  
            for field in threshold_fields:
                if field in raw_controlnet_args:
                    val = raw_controlnet_args[field]
                    if not isinstance(val, int) or val < 0 or val > 255:
                        print(f"⚠️ Invalid ControlNet threshold '{field}' = '{val}', using default 64")
                        raw_controlnet_args[field] = 64
        
        controlnet_args = ControlnetArgs_Dataclass(**raw_controlnet_args) if DATACLASSES_AVAILABLE and hasattr(ControlnetArgs_Dataclass, '__dataclass_fields__') else create_controlnet_args_from_dict(args_dict.get('controlnet_args', {}))

        raw_wan_args = {k: args_dict[k] for k in WanArgs_Config.__dataclass_fields__ if k in args_dict and args_dict[k] is not None} if DATACLASSES_AVAILABLE and hasattr(WanArgs_Config, '__dataclass_fields__') else args_dict.get('wan_args', {}) # Assuming WanArgs_Config is the one from .argument_models
        wan_args = WanArgs_Config(**raw_wan_args) if DATACLASSES_AVAILABLE and hasattr(WanArgs_Config, '__dataclass_fields__') else create_wan_args_from_dict(args_dict.get('wan_args', {})) # Ensure create_wan_args_from_dict is also updated or WanArgs_Config is used directly.
                                                                                                                                                                       # For consistency, direct instantiation is better: WanArgs_Config(**raw_wan_args)
                                                                                                                                                                       # This line might be complex, simplifying for now, but ideally all should use direct instantiation of config.argument_models classes.
        
        # RootArgs handling (remains the same for now, using data_models.RootArgs)
        root_data = {
            'timestring': args_dict.get('timestring', time.strftime('%Y%m%d_%H%M%S')),
            'animation_prompts': args_dict.get('animation_prompts', {"0": "a beautiful landscape"}),
            'models_path': args_dict.get('models_path', 'models'),
            'device': args_dict.get('device', 'cuda'),
            'half_precision': args_dict.get('half_precision', True)
        }
        root = create_root_args_from_dict(root_data)

        # Path processing and other logic needs careful review and adaptation
        # For now, we focus on correct dataclass instantiation.
        # The original logic for constructing outdir in setup_output_directory used args.batch_name and p.outpath_samples.
        # This needs to be done *after* DeforumArgs (aliased as `args`) is created and populated.

        p_obj = args_dict.get('p') # Get the processing object (StableDiffusionProcessingImg2Img instance)
        if p_obj and hasattr(args, 'batch_name') and hasattr(args, 'outdir'): # DeforumArgs has batch_name and outdir defaults
            current_arg_list = [args, anim_args, video_args, parseq_args, loop_args, controlnet_args, wan_args, root]
            # Process batch name with substitutions
            # Assuming args.batch_name is already set by create_deforum_args_from_dict or its defaults
            full_base_folder_path = os.path.join(os.getcwd(), p_obj.outpath_samples)
            processed_batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)
            
            # Construct and set the final outdir on the `args` (DeforumArgs) instance
            final_outdir = os.path.join(p_obj.outpath_samples, str(processed_batch_name))
            final_outdir = os.path.join(os.getcwd(), final_outdir)
            args = dataclasses.replace(args, outdir=os.path.realpath(final_outdir)) # Create new instance with updated outdir
            
            os.makedirs(args.outdir, exist_ok=True)
        elif not hasattr(args, 'outdir') or not args.outdir: # Fallback if p_obj not available or batch_name missing
            args = dataclasses.replace(args, outdir=os.path.join(os.getcwd(), "outputs", "deforum", "fallback"))
            os.makedirs(args.outdir, exist_ok=True)

        log_utils.info(f"Arguments processed successfully with DATACLASSES for index {index}")
        return True, root, args, anim_args, video_args, parseq_args, loop_args, controlnet_args, wan_args
        
    except Exception as e:
        log_utils.error(f"Error processing arguments with DATACLASSES: {e}")
        traceback.print_exc() # Print full traceback for debugging this critical path
        return False, None, None, None, None, None, None, None, None


def validate_args(args, anim_args, video_args):
    """
    Validate argument consistency and requirements.
    Note: FreeU and Kohya HR Fix functionality has been removed.
    """
    errors = []
    
    # Check output directory
    if not hasattr(args, 'outdir') or not args.outdir:
        errors.append("Output directory is required")
    
    # Check frame count
    if not hasattr(anim_args, 'max_frames') or anim_args.max_frames <= 0:
        errors.append("Max frames must be greater than 0")
    
    # Check animation mode
    valid_modes = ['2D', '3D', 'Video Input', 'Interpolation', 'Wan Video']
    if not hasattr(anim_args, 'animation_mode') or anim_args.animation_mode not in valid_modes:
        errors.append(f"Animation mode must be one of: {valid_modes}")
    
    # Check FPS
    if not hasattr(video_args, 'fps') or video_args.fps <= 0:
        errors.append("FPS must be greater than 0")
    
    return errors


def create_default_args():
    """
    Create default argument objects.
    Note: FreeU and Kohya HR Fix functionality has been removed.
    """
    args = SimpleNamespace()
    args.outdir = ""
    args.seed = -1
    args.W = 512
    args.H = 512
    args.steps = 20
    args.cfg_scale = 7.0
    args.sampler = "euler"
    args.scheduler = "normal"
    
    anim_args = SimpleNamespace()
    anim_args.max_frames = 100
    anim_args.animation_mode = "2D"
    anim_args.border = "replicate"
    
    video_args = SimpleNamespace()
    video_args.fps = 15
    video_args.skip_video_creation = False
    
    parseq_args = SimpleNamespace()
    parseq_args.parseq_manifest = ""
    
    loop_args = SimpleNamespace()
    loop_args.use_looper = False
    
    controlnet_args = SimpleNamespace()
    
    wan_args = SimpleNamespace()
    
    return args, anim_args, video_args, parseq_args, loop_args, controlnet_args, wan_args
