from ....generate import generate
from ..flux_controlnet_integration import should_use_flux_controlnet_for_frame, generate_with_flux_controlnet


def call_generate(data, frame: 'DiffusionFrame', redo_seed: int = None):
    # Check if we should use Flux ControlNet for this frame
    if should_use_flux_controlnet_for_frame(data, frame):
        # Use Flux ControlNet generation
        return generate_with_flux_controlnet(data, frame)

    # Standard Forge generation path
    # TODO rename things, data.args.args.strength is actually "denoise", so strength is subtracted from 1.0 when passed.
    ia = data.args
    ia.args.strength = 1.0 - frame.strength  # update denoise for current diffusion from pre-generated frame
    ia.args.seed = frame.seed if redo_seed is None else redo_seed  # update seed with precalculated value from frame
    ia.root.subseed = frame.subseed
    ia.root.subseed_strength = frame.subseed_strength
    index = frame.i - 1
    return generate(ia.args, data.animation_keys.deform_keys, ia.anim_args, ia.loop_args, ia.controlnet_args, ia.root, data.parseq_adapter, index,
                    sampler_name=frame.schedule.sampler_name, scheduler_name=frame.schedule.scheduler_name)
