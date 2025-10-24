import os

from deforum.rendering import filename_helpers as filename_utils
from deforum.rendering import memory as memory_utils
from deforum.depth import DepthModel


def generate_and_save_depth_map_if_active(data, opencv_image, i):
    # Always save depth maps in 3D mode for preview, regardless of save_depth_maps setting
    # They will be cleaned up later if user doesn't want to keep them
    if data.depth_model is not None:
        memory_utils.handle_vram_before_depth_map_generation(data)
        depth = data.depth_model.predict(opencv_image, data.args.anim_args.midas_weight,
                                         data.args.root.half_precision)
        # Ensure depth-maps subdirectory exists
        depth_dir = os.path.join(data.output_directory, "depth-maps")
        os.makedirs(depth_dir, exist_ok=True)

        depth_filename = filename_utils.depth_frame(data, i)
        data.depth_model.save(os.path.join(data.output_directory, depth_filename), depth)
        memory_utils.handle_vram_after_depth_map_generation(data)
        return depth


def create_depth_model_and_enable_depth_map_saving_if_active(anim_mode, root, anim_args, args):
    # Don't override user's save_depth_maps setting - we handle saving and cleanup separately
    return (DepthModel(root.models_path,
                       memory_utils.select_depth_device(root),
                       root.half_precision,
                       keep_in_vram=anim_mode.is_keep_in_vram,
                       depth_algorithm=anim_args.depth_algorithm,
                       Width=args.W, Height=args.H,
                       midas_weight=anim_args.midas_weight)
            if anim_mode.is_predicting_depths else None)
