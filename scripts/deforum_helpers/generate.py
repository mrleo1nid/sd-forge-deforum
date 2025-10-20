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

from PIL import Image
import os
import math
import json
import itertools
import requests
import numexpr
from modules import processing, sd_models
from modules.shared import sd_model, state, cmd_opts
from .deforum_controlnet import is_controlnet_enabled, get_controlnet_script_args
from .prompt import split_weighted_subprompts
from .load_images import load_img, prepare_mask, check_mask_for_errors
from .webui_sd_pipeline import get_webui_sd_pipeline
from .rich import console
from .defaults import get_samplers_list, get_schedulers_list
from .prompt import check_is_number
from .opts_overrider import A1111OptionsOverrider
import cv2
import numpy as np
from types import SimpleNamespace

from .deforum_scripts_overrides import add_forge_script_to_deforum_run, initialise_forge_scripts

from .general_utils import debug_print

def load_mask_latent(mask_input, shape):
    # mask_input (str or PIL Image.Image): Path to the mask image or a PIL Image object
    # shape (list-like len(4)): shape of the image to match, usually latent_image.shape

    if isinstance(mask_input, str):  # mask input is probably a file name
        if mask_input.startswith('http://') or mask_input.startswith('https://'):
            mask_image = Image.open(requests.get(mask_input, stream=True).raw).convert('RGBA')
        else:
            mask_image = Image.open(mask_input).convert('RGBA')
    elif isinstance(mask_input, Image.Image):
        mask_image = mask_input
    else:
        raise Exception("mask_input must be a PIL image or a file name")

    mask_w_h = (shape[-1], shape[-2])
    mask = mask_image.resize(mask_w_h, resample=Image.LANCZOS)
    mask = mask.convert("L")
    return mask

def isJson(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True

# Add pairwise implementation here not to upgrade
# the whole python to 3.10 just for one function
def pairwise_repl(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def generate(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter,  frame=0, sampler_name=None, scheduler_name=None):
    if state.interrupted:
        return None

    if args.reroll_blank_frames == 'ignore':
        return generate_inner(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame, sampler_name, scheduler_name)

    image, caught_vae_exception = generate_with_nans_check(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame, sampler_name, scheduler_name)

    if caught_vae_exception or not image.getbbox():
        patience = args.reroll_patience
        print("Blank frame detected! If you don't have the NSFW filter enabled, this may be due to a glitch!")
        if args.reroll_blank_frames == 'reroll':
            while caught_vae_exception or not image.getbbox():
                print("Rerolling with +1 seed...")
                args.seed += 1
                image, caught_vae_exception = generate_with_nans_check(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame, sampler_name, scheduler_name)
                patience -= 1
                if patience == 0:
                    print("Rerolling with +1 seed failed for 10 iterations! Try setting webui's precision to 'full' and if it fails, please report this to the devs! Interrupting...")
                    state.interrupted = True
                    state.assign_current_image(image)
                    return None
        elif args.reroll_blank_frames == 'interrupt':
            print("Interrupting to save your eyes...")
            state.interrupted = True
            state.assign_current_image(image)
            return None
    return image

def generate_with_nans_check(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame=0, sampler_name=None, scheduler_name=None):
    if cmd_opts.disable_nan_check:
        image = generate_inner(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame, sampler_name, scheduler_name)
    else:
        try:
            image = generate_inner(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame, sampler_name, scheduler_name)
        except Exception as e:
            if "A tensor with all NaNs was produced in VAE." in repr(e):
                print(e)
                return None, True
            else:
                raise e
    return image, False

def generate_inner(args, keys, anim_args, loop_args, controlnet_args,
                   root, parseq_adapter, frame=0, sampler_name=None, scheduler_name=None):
    # Setup the pipeline
    p = get_webui_sd_pipeline(args, root)
    p.prompt, p.negative_prompt = split_weighted_subprompts(args.prompt, frame, anim_args.max_frames)

    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        args.strength = 0
    processed = None
    mask_image = None
    init_image = None
    image_init0 = None
    image_init0_box = None

    if loop_args.use_looper and anim_args.animation_mode in ['2D', '3D']:

        debug_print(f"Looper: use_looper={loop_args.use_looper}, imageStrength={loop_args.imageStrength}, blendFactorMax={loop_args.blendFactorMax}, blendFactorSlope={loop_args.blendFactorSlope}, tweeningFrames={loop_args.tweeningFrameSchedule}, colorCorrectionFactor={loop_args.colorCorrectionFactor}")
        args.strength = loop_args.imageStrength
        tweeningFrames = loop_args.tweeningFrameSchedule
        blendFactor = .07
        colorCorrectionFactor = loop_args.colorCorrectionFactor
        jsonImages = json.loads(loop_args.imagesToKeyframe)
        # find which image to show
        parsedImages = {}
        frameToChoose = 0
        max_f = anim_args.max_frames - 1

        for key, value in jsonImages.items():
            if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
                parsedImages[key] = value
            else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
                parsedImages[int(numexpr.evaluate(key))] = value

        framesToImageSwapOn = list(map(int, list(parsedImages.keys())))

        for swappingFrame in framesToImageSwapOn[1:]:
            frameToChoose += (frame >= int(swappingFrame))

        # find which frame to do our swapping on for tweening
        skipFrame = 25
        for fs, fe in pairwise_repl(framesToImageSwapOn):
            if fs <= frame <= fe:
                skipFrame = fe - fs

        if frame % skipFrame <= tweeningFrames:  # number of tweening frames
            blendFactor = loop_args.blendFactorMax - loop_args.blendFactorSlope * math.cos((frame % tweeningFrames) / (tweeningFrames / 2))
        init_image2, _ = load_img(list(jsonImages.values())[frameToChoose],
                                  None, # init_image_box not used in this case
                                  shape=(args.W, args.H),
                                  use_alpha_as_mask=args.use_alpha_as_mask)
        image_init0 = list(jsonImages.values())[0]

    else:  # they passed in a single init image
        image_init0 = args.init_image
        image_init0_box = args.init_image_box

    available_samplers = get_samplers_list()
    if sampler_name is not None:
        if sampler_name in available_samplers.keys():
            p.sampler_name = available_samplers[sampler_name]
        else:
            raise RuntimeError(f"Sampler name '{sampler_name}' is invalid. Please check the available sampler list in the 'Run' tab")

    available_schedulers = get_schedulers_list()
    if scheduler_name is not None:
        if scheduler_name in available_schedulers.keys():
            p.scheduler_name = available_schedulers[scheduler_name]
        else:
            raise RuntimeError(f"Scheduler name '{scheduler_name}' is invalid. Please check the available scheduler list in the 'Run' tab")

    if args.checkpoint is not None:
        info = sd_models.get_closet_checkpoint_match(args.checkpoint)
        if info is None:
            raise RuntimeError(f"Unknown checkpoint: {args.checkpoint}")
        sd_models.reload_model_weights(info=info)

    if root.init_sample is not None:
        # TODO: cleanup init_sample remains later
        img = root.init_sample
        init_image = img
        if loop_args.use_looper and isJson(loop_args.imagesToKeyframe) and anim_args.animation_mode in ['2D', '3D']:
            init_image = Image.blend(init_image, init_image2, blendFactor)
            correction_colors = Image.blend(init_image, init_image2, colorCorrectionFactor)
            p.color_corrections = [processing.setup_color_correction(correction_colors)]

    # this is the first pass
    elif (loop_args.use_looper and anim_args.animation_mode in ['2D', '3D']) or (args.use_init and ((args.init_image != None and args.init_image != '') or args.init_image_box != None)):
        init_image, mask_image = load_img(image_init0,  # initial init image
                                          image_init0_box,  # initial init image from box (if single init image is used, not json list)
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=args.use_alpha_as_mask)

    else:

        if anim_args.animation_mode != 'Interpolation':
            print(f"Not using an init image (doing pure txt2img)")
        
        if args.motion_preview_mode:
            state.assign_current_image(root.default_img)
            processed = SimpleNamespace(images = [root.default_img], info = "Generating motion preview...")
        else:
            p_txt = processing.StableDiffusionProcessingTxt2Img(
                sd_model=sd_model,
                outpath_samples=root.tmp_deforum_run_duplicated_folder,
                outpath_grids=root.tmp_deforum_run_duplicated_folder,
                prompt=p.prompt,
                styles=p.styles,
                negative_prompt=p.negative_prompt,
                seed=p.seed,
                subseed=p.subseed,
                subseed_strength=p.subseed_strength,
                seed_resize_from_h=p.seed_resize_from_h,
                seed_resize_from_w=p.seed_resize_from_w,
                sampler_name=p.sampler_name,
                scheduler=p.scheduler_name,
                batch_size=p.batch_size,
                n_iter=p.n_iter,
                steps=p.steps,
                cfg_scale=p.cfg_scale,
                distilled_cfg_scale=p.distilled_cfg_scale,
                width=p.width,
                height=p.height,
                restore_faces=p.restore_faces,
                tiling=p.tiling,
                enable_hr=False,
                denoising_strength=0,
            )

            print_combined_table(args, anim_args, p_txt, keys, frame)  # print dynamic table to cli

            initialise_forge_scripts(p_txt)

            if is_controlnet_enabled(controlnet_args):
                cnet_args = get_controlnet_script_args(args, anim_args, controlnet_args, root, parseq_adapter, frame_idx=frame)
                add_forge_script_to_deforum_run(p_txt, "ControlNet", cnet_args)

                debug_print(f"FreeU: {json.dumps(freeu_script_args)}")
                add_forge_script_to_deforum_run(p_txt, "FreeU Integrated", freeu_script_args)

