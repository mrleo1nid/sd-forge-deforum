from modules import sd_schedulers
from modules import sd_samplers


# Dynamically calls all the samplers from forge, so if it updates so does this
def get_samplers_list():
    samplers = {}
    for sampler in sd_samplers.all_samplers:
        samplers[sampler.name.lower()] = sampler.name
    return samplers


def get_schedulers_list():
    return {scheduler.name: scheduler.label for scheduler in sd_schedulers.schedulers}


def get_keyframe_distribution_list():
    return {
        'off': 'Off',
        'keyframes_only': 'Keyframes Only',
        'additive': 'Additive',
        'redistributed': 'Redistributed',
    }


def get_camera_shake_list():
    # Defined in .rendering.data.shakify.shake_data
    return {
        'NONE': 'None',
        'INVESTIGATION': 'Investigation',
        'THE_CLOSEUP': 'The Closeup',
        'THE_WEDDING': 'The Wedding',
        'WALK_TO_THE_STORE': 'Walk to the Store',
        'HANDYCAM_RUN': 'HandyCam Run',
        'OUT_CAR_WINDOW': 'Out Car Window',
        'BIKE_ON_GRAVEL_2D': 'Bike On Gravel (2D)',
        'SPACESHIP_SHAKE_2D': 'Spaceship Shake (2D)',
        'THE_ZEEK_2D': 'The Zeek (2D)',
    }

def DeforumAnimPrompts():
    # Keyframes are synchronized to line up at 60 FPS with amen13 from https://archive.org/details/amen-breaks/:
    # Direct link: https://ia801303.us.archive.org/26/items/amen-breaks/cw_amen13_173.mp3
    return r"""{
        "0": "A sterile hallway, brightly lit with fluorescent lights and empty",
        "12": "A sterile hallway, illuminated and overlooking a construction site through large windows",
        "43": "A sterile hallway, glowing with a digital grid pattern on the walls",
        "74": "An empty parking lot, featuring concrete surfaces and harsh lighting",
        "85": "An empty parking lot, under bright, flickering LED lights",
        "106": "A high-tech facility, with lights flickering in a vast, open area",
        "119": "A cold, reflective surface, illuminated by harsh overhead lights",
        "126": "A sterile environment, with vibrant lights creating a technological ambiance",
        "147": "A sterile space, with cold surfaces reflecting bright lights",
        "158": "A high-tech area, illuminated by neon lights in a clinical setting",
        "178": "A sterile environment, with a sign that says 'Camera Shake in Deforum', attached to a blank wall",
        "210": "A sterile environment, with a sign that says 'Camera Shake in Deforum', attached to a blank wall",
        "241": "A sterile environment, with a sign that says 'Camera Shake in Deforum', attached to a blank wall",
        "262": "An empty space, with a sign that says 'Camera Shake in Deforum', surrounded by intricate mandelbulb fractals on screens",
        "272": "An empty space, with a sign that says 'Camera Shake in Deforum', surrounded by intricate mandelbulb fractals on screens",
        "293": "An empty space, with a sign that says 'Camera Shake in Deforum', surrounded by intricate mandelbulb fractals on screens",
        "314": "An empty space, with a sign that says 'Camera Shake in Deforum', surrounded by intricate mandelbulb fractals on screens",
        "324": "An empty space, with a sign that says 'Camera Shake in Deforum', surrounded by intricate mandelbulb fractals on screens"
    }"""  # WARNING: make sure to not add a trailing semicolon after the last prompt, or the run might break.

def DeforumBunnyPrompts():
    """
    Bunny prompts template synchronized to amen break at 60 FPS.
    These are the popular default prompts showcasing cyberpunk bunny progression.
    """
    return r"""{
        "0": "A cute bunny, hopping on grass, photorealistic",
        "12": "A cute bunny with sunglasses, hopping at a neon-lit construction site",
        "43": "A cyberpunk bunny with glowing eyes, standing on a digital grid, retrowave aesthetic",
        "74": "A cool anthropomorphic bunny in a leather jacket, mounting a futuristic motorcycle",
        "85": "A badass synthwave bunny with neon mohawk, riding a glowing hoverbike through a cyberpunk city",
        "106": "A cool synthwave bunny in metallic armor, riding a motorcycle with flaming wheels across burning coal",
        "119": "A synthwave bunny with mirrored visor helmet, riding a cryogenic ice motorcycle across a frozen lake, digital horizon",
        "126": "A synthwave bunny with laser eyes, motorcycle transforming into a fire-breathing machine, burning coal road, purple horizon",
        "147": "A neon-outlined synthwave bunny, motorcycle creating ice crystals, racing across a frozen digital wasteland, blue glow",
        "158": "A synthwave bunny with holographic jacket, riding a dimensional-shifting motorcycle through lava fields, synthwave sunset",
        "178": "A cool synthwave bunny with robotic arm, hovering motorcycle, holding a neon sign that says 'Deforum & Forge'",
        "210": "A synthwave bunny DJ with glowing headphones, motorcycle parked nearby, raising a holographic sign that says 'Deforum & Forge'",
        "241": "A synthwave cyborg bunny with visor shades, futuristic motorcycle morphing into a digital throne, neon sign says 'Deforum & Forge'",
        "262": "A transcendent synthwave bunny with energy aura, quantum motorcycle, surrounded by mandelbulb fractals, holding a sign that says 'Deforum & Forge'",
        "272": "A godlike synthwave bunny, digital motorcycle breaking into particle effects, mandelbulb fractals forming reality portals, sign that says 'Deforum & Forge'",
        "293": "A synthwave bunny in virtual space, motorcycle trails leaving data streams, kaleidoscopic mandelbulb fractals, sign that says 'Deforum & Forge'",
        "314": "A synthwave bunny becoming one with the digital realm, motorcycle dissolving into the fractal patterns, mandelbulb universe, sign that says 'Deforum & Forge'",
        "324": "An ascended synthwave bunny deity, motorcycle transformed into throne of light, ruling over an empire of mandelbulb fractals, glowing sign that says 'Deforum & Forge'"
    }"""

def DeforumWanPrompts():
    """
    Wan video prompts template - drip progression theme.
    """
    return r"""{
        "0": "A cute white bunny sitting in a peaceful meadow, soft natural lighting, photorealistic",
        "12": "A white bunny with slightly glowing fur, sitting in a meadow with subtle magical sparkles",
        "43": "A bunny with soft neon highlights on its fur, sitting in a meadow with digital aurora effects",
        "74": "A bunny with cyberpunk fur patterns, glowing blue and purple, in an urban meadow setting",
        "85": "A bunny with LED-trimmed ears and glowing whiskers, cyberpunk aesthetic, neon city background",
        "106": "A tech bunny with holographic fur patterns, sitting in a futuristic garden environment",
        "119": "A bunny wearing sleek chrome accessories, reflective metallic fur highlights, sci-fi setting",
        "126": "A drip bunny with golden chains and designer accessories, confident pose, luxury environment",
        "147": "A swag bunny with diamond earrings and platinum fur trim, posing with attitude",
        "158": "A boss bunny with bling accessories and designer sunglasses, standing confidently",
        "178": "A supreme drip bunny with ice-cold chains, golden grillz, and designer everything",
        "210": "A legendary bunny deity with cosmic bling, floating in space with stellar accessories",
        "241": "An ascended bunny with celestial drip, surrounded by floating diamonds and gold",
        "262": "A transcendent bunny overlord with reality-bending bling, fractal jewelry patterns",
        "272": "A hyperdimensional drip bunny with impossible geometry accessories, glowing with power",
        "293": "An omnipotent bunny god with universal bling, commanding cosmic forces",
        "314": "A supreme bunny entity with reality-warping drip, existing beyond time and space",
        "324": "The ultimate drip bunny, transcending all dimensions with infinite swag and cosmic bling"
    }"""

def get_default_settings_template(template_type="bunny"):
    """
    Create default settings with specified prompt template.
    
    Args:
        template_type: "bunny", "sterile", "wan", or "minimal"
        
    Returns:
        dict: Complete default settings dictionary
    """
    # Base settings - these are common to all templates
    base_settings = {
        "W": 1280,
        "H": 720,
        "show_info_on_ui": True,
        "tiling": False,
        "restore_faces": False,
        "seed_resize_from_w": 0,
        "seed_resize_from_h": 0,
        "seed": -1,
        "sampler": "Euler",
        "scheduler": "Simple",
        "steps": 20,
        "batch_name": "Deforum_{timestring}",
        "keyframe_distribution": "Keyframes Only",
        "seed_behavior": "iter",
        "seed_iter_N": 1,
        "use_init": False,
        "strength": 0.85,
        "strength_0_no_init": True,
        "init_image": None,
        "use_mask": False,
        "use_alpha_as_mask": False,
        "mask_file": "https://deforum.github.io/a1/M1.jpg",
        "invert_mask": False,
        "mask_contrast_adjust": 1.0,
        "mask_brightness_adjust": 1.0,
        "overlay_mask": True,
        "mask_overlay_blur": 4,
        "fill": 0,
        "full_res_mask": True,
        "full_res_mask_padding": 4,
        "reroll_blank_frames": "ignore",
        "reroll_patience": 10.0,
        "motion_preview_mode": False,
        "animation_mode": "3D",
        "max_frames": 334,
        "border": "wrap",
        "angle": "0: (0)",
        "zoom": "0: (1.0025+0.002*sin(1.25*3.14*t/120))",
        "translation_x": "0: (0)",
        "translation_y": "0: (0)",
        "translation_z": "0: (1.0)",
        "transform_center_x": "0: (0.5)",
        "transform_center_y": "0: (0.5)",
        "rotation_3d_x": "0: (0)",
        "rotation_3d_y": "0: (0)",
        "rotation_3d_z": "0: (0)",
        "shake_name": "INVESTIGATION",
        "shake_intensity": 1.0,
        "shake_speed": 1.0,
        "enable_perspective_flip": False,
        "perspective_flip_theta": "0: (0)",
        "perspective_flip_phi": "0: (0)",
        "perspective_flip_gamma": "0: (0)",
        "perspective_flip_fv": "0: (53)",
        "noise_schedule": "0: (0.065)",
        "strength_schedule": "0: (0.85)",
        "keyframe_strength_schedule": "0: (0.20)",
        "contrast_schedule": "0: (1.0)",
        "cfg_scale_schedule": "0: (1.0)",
        "distilled_cfg_scale_schedule": "0: (3.5)",
        "enable_steps_scheduling": False,
        "steps_schedule": "0: (20)",
        "fov_schedule": "0: (70)",
        "aspect_ratio_schedule": "0: (1.0)",
        "aspect_ratio_use_old_formula": False,
        "near_schedule": "0: (200)",
        "far_schedule": "0: (10000)",
        "seed_schedule": "0:(s), 1:(-1), \"max_f-2\":(-1), \"max_f-1\":(s)",
        "enable_subseed_scheduling": False,
        "subseed_schedule": "0: (1)",
        "subseed_strength_schedule": "0: (0)",
        "enable_sampler_scheduling": False,
        "sampler_schedule": "0: (\"Euler\")",
        "enable_scheduler_scheduling": False,
        "scheduler_schedule": "0: (\"Simple\")",
        "use_noise_mask": False,
        "mask_schedule": "0: (\"{video_mask}\")",
        "noise_mask_schedule": "0: (\"{video_mask}\")",
        "enable_checkpoint_scheduling": False,
        "checkpoint_schedule": "0: (\"model1.ckpt\"), 100: (\"model2.safetensors\")",
        "enable_clipskip_scheduling": False,
        "clipskip_schedule": "0: (2)",
        "enable_noise_multiplier_scheduling": True,
        "noise_multiplier_schedule": "0: (1.05)",
        "resume_from_timestring": False,
        "resume_timestring": "20251111111111",
        "enable_ddim_eta_scheduling": False,
        "ddim_eta_schedule": "0: (0)",
        "enable_ancestral_eta_scheduling": False,
        "ancestral_eta_schedule": "0: (1.0)",
        "amount_schedule": "0: (0.1)",
        "kernel_schedule": "0: (5)",
        "sigma_schedule": "0: (1)",
        "threshold_schedule": "0: (0)",
        "color_coherence": "None",
        "color_coherence_image_path": "https://upload.wikimedia.org/wikipedia/commons/7/72/Grautoene.png",
        "color_coherence_video_every_N_frames": 1,
        "color_force_grayscale": False,
        "legacy_colormatch": False,
        "diffusion_cadence": 10,
        "optical_flow_cadence": "None",
        "cadence_flow_factor_schedule": "0: (1)",
        "optical_flow_redo_generation": "None",
        "redo_flow_factor_schedule": "0: (1)",
        "diffusion_redo": "0",
        "noise_type": "perlin",
        "perlin_octaves": 4,
        "perlin_persistence": 0.5,
        "use_depth_warping": True,
        "depth_algorithm": "Depth-Anything-V2-small",
        "midas_weight": 0.2,
        "padding_mode": "border",
        "sampling_mode": "bicubic",
        "save_depth_maps": False,
        "video_init_path": "https://deforum.github.io/a1/V1.mp4",
        "extract_nth_frame": 1,
        "extract_from_frame": 0,
        "extract_to_frame": -1,
        "overwrite_extracted_frames": False,
        "use_mask_video": False,
        "video_mask_path": "https://deforum.github.io/a1/VM1.mp4",
        "parseq_manifest": "",
        "parseq_use_deltas": True,
        "parseq_non_schedule_overrides": True,
        "use_looper": False,
        "init_images": get_guided_imgs_default_json(),
        "image_strength_schedule": "0:(0.85)",
        "image_keyframe_strength_schedule": "0:(0.20)",
        "blendFactorMax": "0:(0.35)",
        "blendFactorSlope": "0:(0.25)",
        "tweening_frames_schedule": "0:(20)",
        "color_correction_factor": "0:(0.075)",
        "positive_prompts": "",
        "negative_prompts": "",
        # ControlNet settings
        **_get_controlnet_defaults(),
        # Wan settings  
        **_get_wan_defaults(),
        # Output settings
        "skip_video_creation": False,
        "fps": 60,
        "make_gif": False,
        "delete_imgs": False,
        "delete_input_frames": False,
        "add_soundtrack": "File",
        "soundtrack_path": "https://ia801303.us.archive.org/26/items/amen-breaks/cw_amen13_173.mp3",
        "r_upscale_video": False,
        "r_upscale_factor": "x2",
        "r_upscale_model": "realesr-animevideov3",
        "r_upscale_keep_imgs": True,
        "store_frames_in_ram": False,
        "frame_interpolation_engine": "None",
        "frame_interpolation_x_amount": 2,
        "frame_interpolation_slow_mo_enabled": False,
        "frame_interpolation_slow_mo_amount": 2,
        "frame_interpolation_keep_imgs": True,
        "frame_interpolation_use_upscaled": False,
        "sd_model_name": "Flux\\flux1-dev-bnb-nf4-v2.safetensors",
        "sd_model_hash": "f0770152",
        "deforum_git_commit_id": "Unknown"
    }
    
    # Add prompts based on template type
    import json
    if template_type == "bunny":
        base_settings["prompts"] = json.loads(DeforumBunnyPrompts())
        base_settings["wan_prompts"] = json.loads(DeforumWanPrompts())
    elif template_type == "sterile":
        base_settings["prompts"] = json.loads(DeforumAnimPrompts())
        base_settings["wan_prompts"] = {"0": "A sterile environment, minimalist and clean"}
    elif template_type == "wan":
        base_settings["prompts"] = {"0": "A simple scene for video generation"}
        base_settings["wan_prompts"] = json.loads(DeforumWanPrompts())
    else:  # minimal
        base_settings["prompts"] = {"0": "A beautiful landscape"}
        base_settings["wan_prompts"] = {"0": "A simple video scene"}
    
    return base_settings

def _get_controlnet_defaults():
    """Get default ControlNet settings."""
    defaults = {}
    for i in range(1, 6):
        defaults.update({
            f"cn_{i}_overwrite_frames": True,
            f"cn_{i}_vid_path": "",
            f"cn_{i}_mask_vid_path": "",
            f"cn_{i}_enabled": False,
            f"cn_{i}_low_vram": False,
            f"cn_{i}_pixel_perfect": True,
            f"cn_{i}_module": "none",
            f"cn_{i}_model": "None",
            f"cn_{i}_weight": "0:(1)",
            f"cn_{i}_guidance_start": "0:(0.0)",
            f"cn_{i}_guidance_end": "0:(1.0)",
            f"cn_{i}_processor_res": 64,
            f"cn_{i}_threshold_a": 64,
            f"cn_{i}_threshold_b": 64,
            f"cn_{i}_resize_mode": "Inner Fit (Scale to Fit)",
            f"cn_{i}_control_mode": "Balanced",
            f"cn_{i}_loopback_mode": True,
        })
    return defaults

def _get_wan_defaults():
    """Get default Wan settings."""
    return {
        "wan_t2v_model": "1.3B VACE",
        "wan_i2v_model": "Use Primary Model",
        "wan_auto_download": True,
        "wan_preferred_size": "1.3B VACE (Recommended)",
        "wan_model_path": "models/wan",
        "wan_resolution": "864x480",
        "wan_seed": -1,
        "wan_inference_steps": 20,
        "wan_guidance_scale": 7.5,
        "wan_strength_override": True,
        "wan_fixed_strength": 1.0,
        "wan_guidance_override": True,
        "wan_frame_overlap": 2,
        "wan_motion_strength": 1.0,
        "wan_enable_interpolation": True,
        "wan_interpolation_strength": 0.5,
        "wan_flash_attention_mode": "Auto (Recommended)",
    }

# Guided images defaults
def get_guided_imgs_default_json():
    return '''{
    "0": "https://deforum.github.io/a1/Gi1.png",
    "max_f/4-5": "https://deforum.github.io/a1/Gi2.png",
    "max_f/2-10": "https://deforum.github.io/a1/Gi3.png",
    "3*max_f/4-15": "https://deforum.github.io/a1/Gi4.jpg",
    "max_f-20": "https://deforum.github.io/a1/Gi1.png"
}'''

def get_wan_video_info_html():
    return """
        <p style="padding-bottom:0">
            <b style="text-shadow: blue -1px -1px;">Wan 2.1 Video Generation</b>
            <span style="color:#DDD;font-size:0.7rem;text-shadow: black -1px -1px;margin-left:10px;">
                powered by <a href="https://github.com/Wan-Video/Wan2.1" target="_blank">Wan 2.1</a>
            </span>
        </p>
        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em;">
            <li><b>Text-to-Video</b>: Generate video clips directly from text prompts using Wan 2.1</li>
            <li><b>Image-to-Video</b>: Continue video generation using the last frame of the previous clip as initialization</li>
            <li><b>Frame Continuity</b>: Seamless transitions between clips ensure smooth video flow</li>
            <li><b>Audio Synchronization</b>: Align video clips with audio timing for perfect synchronization</li>
            <li><b>Prompt Scheduling</b>: Use Deforum's prompt scheduling system with frame-based timing</li>
        </ul>
        
        <p><b>Setup Requirements:</b></p>
        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em;">
            <li>Download and install <a href="https://github.com/Wan-Video/Wan2.1" target="_blank">Wan 2.1</a></li>
            <li>Set the correct model path in the Wan Model Path field</li>
            <li>Ensure you have sufficient GPU memory (recommended: 12GB+ VRAM)</li>
            <li>Configure your prompts using standard Deforum JSON format</li>
        </ul>
        
        <p><b>How It Works:</b></p>
        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em;">
            <li><b>First Clip</b>: Generated using text-to-video from your first prompt</li>
            <li><b>Subsequent Clips</b>: Generated using image-to-video with the last frame as init image</li>
            <li><b>Timing</b>: Clip duration calculated from prompt frame positions and FPS settings</li>
            <li><b>Output</b>: All clips are stitched together using FFmpeg for final video</li>
        </ul>
        
        <p><b>Performance Tips:</b></p>
        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em;">
            <li>Use shorter clip durations (2-4 seconds) for better memory efficiency</li>
            <li>Lower inference steps (20-30) for faster generation</li>
            <li>Start with 512x512 resolution for testing, scale up for final renders</li>
            <li>Enable frame overlap for smoother transitions between clips</li>
        </ul>
        
        <p><b>Limitations:</b></p>
        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em;">
            <li>Traditional Deforum camera movements are disabled in Wan mode</li>
            <li>3D depth warping and optical flow are not compatible</li>
            <li>Generation time is significantly longer than traditional diffusion</li>
            <li>Requires high-end hardware for optimal performance</li>
        </ul>
        
        <a style='color:SteelBlue;' target='_blank' href='https://github.com/Wan-Video/Wan2.1'>Visit Wan 2.1 Repository</a> for more information and installation instructions.
        """

def get_composable_masks_info_html():
    return """
        <ul style="list-style-type:circle; margin-left:0.75em; margin-bottom:0.2em">
        <li>To enable, check use_mask in the Init tab</li>
        <li>Supports boolean operations: (! - negation, & - and, | - or, ^ - xor, \\\\ - difference, () - nested operations)</li>
        <li>default variables: in \\{\\}, like \\{init_mask\\}, \\{video_mask\\}, \\{everywhere\\}</li>
        <li>masks from files: in [], like [mask1.png]</li>
        <li>description-based: <i>word masks</i> in &lt;&gt;, like &lt;apple&gt;, &lt;hair&gt</li>
        </ul>
        """
        
def get_parseq_info_html():
    return """
        <p>Use a <a style='color:SteelBlue;' target='_blank' href='https://sd-parseq.web.app/deforum'>Parseq</a> manifest for your animation (leave blank to ignore).</p>
        <p style="margin-top:1em; margin-bottom:1em;">
            Fields managed in your Parseq manifest override the values and schedules set in other parts of this UI. You can select which values to override by using the "Managed Fields" section in Parseq.
        </p>
        """
        
def get_prompts_info_html():
    return """
        <ul style="list-style-type:circle; margin-left:0.75em; margin-bottom:0.2em">
        <li>Please always keep values in math functions above 0.</li>
        <li>There is *no* Batch mode like in vanilla deforum. Please Use the txt2img tab for that.</li>
        <li>For negative prompts, please write your positive prompt, then --neg ugly, text, assymetric, or any other negative tokens of your choice. OR:</li>
        <li>Use the negative_prompts field to automatically append all words as a negative prompt. *Don't* add --neg in the negative_prompts field!</li>
        <li>Prompts are stored in JSON format. If you've got an error, check it in a <a style="color:SteelBlue" href="https://odu.github.io/slingjsonlint/">JSON Validator</a></li>
        </ul>
        """
        
def get_guided_imgs_info_html():
    return """        
        <p>You can use this as a guided image tool or as a looper depending on your settings in the keyframe images field. 
        Set the keyframes and the images that you want to show up. 
        Note: the number of frames between each keyframe should be greater than the tweening frames.</p>

        <p>Prerequisites and Important Info:</p>
        <ul style="list-style-type:circle; margin-left:2em; margin-bottom:0em">
            <li>This mode works ONLY with 2D/3D animation modes. Interpolation and Video Input modes aren't supported.</li>
            <li>Init tab's strength slider should be greater than 0. Recommended value (.65 - .80).</li>
            <li>'seed_behavior' will be forcibly set to 'schedule'.</li>
        </ul>
        
        <p>Looping recommendations:</p>
        <ul style="list-style-type:circle; margin-left:2em; margin-bottom:0em">
            <li>seed_schedule should start and end on the same seed.<br />
            Example: seed_schedule could use 0:(5), 1:(-1), 219:(-1), 220:(5)</li>
            <li>The 1st and last keyframe images should match.</li>
            <li>Set your total number of keyframes to be 21 more than the last inserted keyframe image.<br />
            Example: Default args should use 221 as the total keyframes.</li>
            <li>Prompts are stored in JSON format. If you've got an error, check it in the validator, 
            <a style="color:SteelBlue" href="https://odu.github.io/slingjsonlint/">like here</a></li>
        </ul>
        
        <p>The Guided images mode exposes the following variables for the prompts and the schedules:</p>
        <ul style="list-style-type:circle; margin-left:2em; margin-bottom:0em">
            <li><b>s</b> is the <i>initial</i> seed for the whole video generation.</li>
            <li><b>max_f</b> is the length of the video, in frames.<br />
            Example: seed_schedule could use 0:(s), 1:(-1), "max_f-2":(-1), "max_f-1":(s)</li>
            <li><b>t</b> is the current frame number.<br />
            Example: strength_schedule could use 0:(0.25 * cos((72 / 60 * 3.141 * (t + 0) / 30))**13 + 0.7) to make alternating changes each 30 frames</li>
        </ul>
        """
        
def get_main_info_html():
    return """
        <p><strong>Made by <a href="https://deforum.github.io">deforum.github.io</a>, fork for WebUI Forge maintained by <a href="https://github.com/Tok/sd-forge-deforum">Zirteq</a>.</strong></p>
        <p><a  style="color:SteelBlue" href="https://github.com/Tok/sd-forge-deforum/wiki/FAQ-&-Troubleshooting">FOR HELP CLICK HERE</a></p>
        <ul style="list-style-type:circle; margin-left:1em">
        <li>The code for this fork: <a  style="color:SteelBlue" href="https://github.com/Tok/sd-forge-deforum">here</a>.</li>
        <li>Join the <a style="color:SteelBlue" href="https://discord.gg/deforum">official Deforum Discord</a> to share your creations and suggestions.</li>
        <li>Original Deforum Wiki: <a style="color:SteelBlue" href="https://github.com/deforum-art/deforum-for-automatic1111-webui/wiki">here</a>.</li>
        <li>Anime-inclined great guide (by FizzleDorf) with lots of examples: <a style="color:SteelBlue" href="https://rentry.org/AnimAnon-Deforum">here</a>.</li>
        <li>For advanced keyframing with Math functions, see <a style="color:SteelBlue" href="https://github.com/deforum-art/deforum-for-automatic1111-webui/wiki/Maths-in-Deforum">here</a>.</li>
        <li>Alternatively, use <a style="color:SteelBlue" href="https://sd-parseq.web.app/deforum">sd-parseq</a> as a UI to define your animation schedules (see the Parseq section in the Init tab).</li>
        <li><a style="color:SteelBlue" href="https://www.framesync.xyz/">framesync.xyz</a> is also a good option, it makes compact math formulae for Deforum keyframes by selecting various waveforms.</li>
        <li>The other site allows for making keyframes using <a style="color:SteelBlue" href="https://www.chigozie.co.uk/keyframe-string-generator/">interactive splines and Bezier curves</a> (select Disco output format).</li>
        <li>If you want to use Width/Height which are not multiples of 64, please change noise_type to 'Uniform', in Keyframes --> Noise.</li>
        </ul>
        <italic>If you liked this fork, please <a style="color:SteelBlue" href="https://github.com/Tok/sd-forge-deforum">give it a star on GitHub</a>!</italic> 😊
        """
def get_frame_interpolation_info_html():
    return """
        Use <a href="https://github.com/megvii-research/ECCV2022-RIFE">RIFE</a> / <a href="https://film-net.github.io/">FILM</a> Frame Interpolation to smooth out, slow-mo (or both) any video.</p>
         <p style="margin-top:1em">
            Supported engines:
            <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em">
                <li>RIFE v4.6 and FILM.</li>
            </ul>
        </p>
         <p style="margin-top:1em">
            Important notes:
            <ul style="list-style-type:circle; margin-left:1em; margin-bottom:1em">
                <li>Frame Interpolation will *not* run if any of the following are enabled: 'Store frames in ram' / 'Skip video for run all'.</li>
                <li>Audio (if provided) will *not* be transferred to the interpolated video if Slow-Mo is enabled.</li>
                <li>'add_soundtrack' and 'soundtrack_path' aren't being honoured in "Interpolate an existing video" mode. Original vid audio will be used instead with the same slow-mo rules above.</li>
                <li>In "Interpolate existing pics" mode, FPS is determined *only* by output FPS slider. Audio will be added if requested even with slow-mo "enabled", as it does *nothing* in this mode.</li>
            </ul>
        </p>
        """
def get_frames_to_video_info_html():
    return """
        <p style="margin-top:0em">
        Important Notes:
        <ul style="list-style-type:circle; margin-left:1em; margin-bottom:0.25em">
            <li>Enter relative to webui folder or Full-Absolute path, and make sure it ends with something like this: '20230124234916_%09d.png', just replace 20230124234916 with your batch ID. The %09d is important, don't forget it!</li>
            <li>In the filename, '%09d' represents the 9 counting numbers, For '20230124234916_000000001.png', use '20230124234916_%09d.png'</li>
            <li>If non-deforum frames, use the correct number of counting digits. For files like 'bunnies-0000.jpg', you'd use 'bunnies-%04d.jpg'</li>
        </ul>
        """
def get_leres_info_html():
    return 'Note that LeReS has a Non-Commercial <a href="https://github.com/aim-uofa/AdelaiDepth/blob/main/LeReS/LICENSE" target="_blank">license</a>. Use it only for fun/personal use.'

def get_gradio_html(section_name):
    # Note: hybrid_video section removed - functionality not available
    if section_name.lower() == 'wan_video':
        return get_wan_video_info_html()
    elif section_name.lower() == 'composable_masks':
        return get_composable_masks_info_html()
    elif section_name.lower() == 'parseq':
        return get_parseq_info_html()
    elif section_name.lower() == 'prompts':
        return get_prompts_info_html()
    elif section_name.lower() == 'guided_imgs':
        return get_guided_imgs_info_html()
    elif section_name.lower() == 'main':
        return get_main_info_html()
    elif section_name.lower() == 'frame_interpolation':
        return get_frame_interpolation_info_html()
    elif section_name.lower() == 'frames_to_video':
        return get_frames_to_video_info_html()
    elif section_name.lower() == 'leres':
        return get_leres_info_html()
    else:
        return ""

mask_fill_choices = ['fill', 'original', 'latent noise', 'latent nothing']
        
