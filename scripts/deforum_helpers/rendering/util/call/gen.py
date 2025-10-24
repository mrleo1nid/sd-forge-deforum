from ....generate import generate
from ..flux_controlnet_integration import should_use_flux_controlnet_for_frame, prepare_flux_controlnet_for_frame
from ....flux_controlnet_forge_injection import clear_control_samples


def call_generate(data, frame: 'DiffusionFrame', redo_seed: int = None):
    # Check if we should use Flux ControlNet for this frame
    if should_use_flux_controlnet_for_frame(data, frame):
        # Prepare Flux ControlNet control samples (V2 - Forge-native)
        print(f"🌐 Using Flux ControlNet V2 for keyframe {frame.i}")
        prepare_flux_controlnet_for_frame(data, frame)
        # Control samples are now stored and will be injected into Forge's pipeline
        # Continue with standard generation flow

    # Standard Forge generation path (now includes Flux ControlNet if enabled)
    # TODO rename things, data.args.args.strength is actually "denoise", so strength is subtracted from 1.0 when passed.
    ia = data.args
    ia.args.strength = 1.0 - frame.strength  # update denoise for current diffusion from pre-generated frame
    ia.args.seed = frame.seed if redo_seed is None else redo_seed  # update seed with precalculated value from frame
    ia.root.subseed = frame.subseed
    ia.root.subseed_strength = frame.subseed_strength
    index = frame.i - 1

    try:
        result = generate(ia.args, data.animation_keys.deform_keys, ia.anim_args, ia.loop_args, ia.controlnet_args, ia.root, data.parseq_adapter, index,
                        sampler_name=frame.schedule.sampler_name, scheduler_name=frame.schedule.scheduler_name)
        return result
    finally:
        # Clear control samples after generation
        clear_control_samples()
