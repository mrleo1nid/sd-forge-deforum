"""
Animation Components
Contains animation controls and settings
"""

import gradio as gr
from modules.ui_components import FormRow, FormColumn, ToolButton
from .component_builders import create_gr_elem, create_row
from ..utils import emoji_utils


def get_tab_animation(da, dloopArgs):
    """Create the Animation tab with animation-specific controls.
    
    Args:
        da: DeforumAnimArgs instance
        dloopArgs: LoopArgs instance
        
    Returns:
        Dict of component references
    """
    with gr.TabItem(f"{emoji_utils.movie_camera()} Animation"):
        
        # PROMPTS SECTION
        with gr.Accordion("📝 Animation Prompts", open=True):
            gr.Markdown("""
            **Animation Prompts:** Define what happens at each keyframe of your animation.
            Use frame numbers to specify when prompts should change.
            """)
            
            with FormRow():
                animation_prompts = gr.Textbox(
                    label="Animation prompts",
                    lines=12,
                    value="",
                    interactive=True,
                    elem_id="animation_prompts",
                    info="JSON format: {\"0\": \"prompt for frame 0\", \"60\": \"prompt for frame 60\"}"
                )
                
            with FormRow():
                animation_prompts_positive = gr.Textbox(
                    label="Positive prompts",
                    lines=4,
                    value="",
                    interactive=True,
                    elem_id="animation_prompts_positive",
                    visible=False
                )
                
            with FormRow():
                animation_prompts_negative = gr.Textbox(
                    label="Negative prompts", 
                    lines=4,
                    value="",
                    interactive=True,
                    elem_id="animation_prompts_negative",
                    visible=False
                )
        
        # ANIMATION SETTINGS
        with gr.Accordion("⚙️ Animation Settings", open=True):
            with FormRow():
                animation_mode = create_gr_elem(da.animation_mode)
                border = create_gr_elem(da.border)
                
            with FormRow():
                max_frames = create_gr_elem(da.max_frames)
                diffusion_cadence = create_gr_elem(da.diffusion_cadence)
                
        # VIDEO INPUT SETTINGS
        with gr.Accordion("📹 Video Input", open=False):
            with FormRow():
                video_init_path = create_gr_elem(da.video_init_path)
                
            with FormRow():
                extract_from_frame = create_gr_elem(da.extract_from_frame)
                extract_to_frame = create_gr_elem(da.extract_to_frame)
                
            with FormRow():
                extract_nth_frame = create_gr_elem(da.extract_nth_frame)
                overwrite_extracted_frames = create_gr_elem(da.overwrite_extracted_frames)
                
            with FormRow():
                use_mask_video = create_gr_elem(da.use_mask_video)
                video_mask_path = create_gr_elem(da.video_mask_path)
        
        # GUIDED IMAGES SECTION (moved from keyframes)
        with gr.Accordion("🖼️ Guided Images", open=False):
            gr.Markdown("""
            **Guided Images Mode:** Use a sequence of images to guide the animation.
            Each image will influence the generation at its corresponding frame.
            """)
            
            with FormRow():
                use_looper = create_gr_elem(dloopArgs.use_looper)
                
            with FormRow():
                init_images = create_gr_elem(dloopArgs.init_images)
                
            # Guided Images Schedules
            with gr.Accordion("📊 Guided Images Schedules", open=False):
                with FormRow():
                    image_strength_schedule = create_gr_elem(dloopArgs.image_strength_schedule)
                    
                with FormRow():
                    image_keyframe_strength_schedule = create_gr_elem(dloopArgs.image_keyframe_strength_schedule)
                    
                with FormRow():
                    blendFactorMax = create_gr_elem(dloopArgs.blendFactorMax)
                    blendFactorSlope = create_gr_elem(dloopArgs.blendFactorSlope)
                    
                with FormRow():
                    tweening_frames_schedule = create_gr_elem(dloopArgs.tweening_frames_schedule)
                    color_correction_factor = create_gr_elem(dloopArgs.color_correction_factor)
        
        # RESUME ANIMATION
        with gr.Accordion("🔄 Resume Animation", open=False):
            with FormRow():
                resume_from_timestring = create_gr_elem(da.resume_from_timestring)
                resume_timestring = create_gr_elem(da.resume_timestring)
        
    # Return only the actual UI components created in this function
    return {
        'animation_mode': animation_mode,
        'max_frames': max_frames,
        'border': border,
        'video_init_path': video_init_path,
        'extract_from_frame': extract_from_frame,
        'extract_to_frame': extract_to_frame,
        'extract_nth_frame': extract_nth_frame,
        'overwrite_extracted_frames': overwrite_extracted_frames,
        'use_mask_video': use_mask_video,
        'video_mask_path': video_mask_path,
        'use_looper': use_looper,
        'init_images': init_images,
        'image_strength_schedule': image_strength_schedule,
        'image_keyframe_strength_schedule': image_keyframe_strength_schedule,
        'blendFactorMax': blendFactorMax,
        'blendFactorSlope': blendFactorSlope,
        'tweening_frames_schedule': tweening_frames_schedule,
        'color_correction_factor': color_correction_factor,
        'resume_from_timestring': resume_from_timestring,
        'resume_timestring': resume_timestring,
    }


def get_tab_prompts(da):
    """Create the Prompts tab with detailed prompt editing.
    
    Args:
        da: DeforumAnimArgs instance
        
    Returns:
        Dict of component references
    """
    with gr.TabItem(f"{emoji_utils.pencil()} Prompts"):
        gr.Markdown("""
        ## Animation Prompts
        
        Define what happens at each keyframe of your animation. Use JSON format to specify 
        which prompts should be active at which frames.
        
        **Format:** `{"0": "first prompt", "60": "second prompt", "120": "third prompt"}`
        """)
        
        # Main prompts editor
        with FormRow():
            animation_prompts = gr.Textbox(
                label="Animation prompts",
                lines=15,
                value='{\n    "0": "A cute bunny, hopping on grass, photorealistic",\n    "12": "A cute bunny with sunglasses, hopping at a neon-lit construction site",\n    "43": "A cyberpunk bunny with glowing eyes, standing on a digital grid, retrowave aesthetic",\n    "74": "A cool anthropomorphic bunny in a leather jacket, mounting a futuristic motorcycle",\n    "85": "A badass synthwave bunny with neon mohawk, riding a glowing hoverbike through a cyberpunk city",\n    "106": "A cool synthwave bunny in metallic armor, riding a motorcycle with flaming wheels across burning coal",\n    "119": "A synthwave bunny with mirrored visor helmet, riding a cryogenic ice motorcycle across a frozen lake, digital horizon",\n    "126": "A synthwave bunny with laser eyes, motorcycle transforming into a fire-breathing machine, burning coal road, purple horizon",\n    "147": "A neon-outlined synthwave bunny, motorcycle creating ice crystals, racing across a frozen digital wasteland, blue glow",\n    "158": "A synthwave bunny with holographic jacket, riding a dimensional-shifting motorcycle through lava fields, synthwave sunset",\n    "178": "A cool synthwave bunny with robotic arm, hovering motorcycle, holding a neon sign that says \'Deforum & Forge\'",\n    "210": "A synthwave bunny DJ with glowing headphones, motorcycle parked nearby, raising a holographic sign that says \'Deforum & Forge\'",\n    "241": "A synthwave cyborg bunny with visor shades, futuristic motorcycle morphing into a digital throne, neon sign says \'Deforum & Forge\'",\n    "262": "A transcendent synthwave bunny with energy aura, quantum motorcycle, surrounded by mandelbulb fractals, holding a sign that says \'Deforum & Forge\'",\n    "272": "A godlike synthwave bunny, digital motorcycle breaking into particle effects, mandelbulb fractals forming reality portals, sign that says \'Deforum & Forge\'",\n    "293": "A synthwave bunny in virtual space, motorcycle trails leaving data streams, kaleidoscopic mandelbulb fractals, sign that says \'Deforum & Forge\'",\n    "314": "A synthwave bunny becoming one with the digital realm, motorcycle dissolving into the fractal patterns, mandelbulb universe, sign that says \'Deforum & Forge\'",\n    "324": "An ascended synthwave bunny deity, motorcycle transformed into throne of light, ruling over an empire of mandelbulb fractals, glowing sign that says \'Deforum & Forge\'"\n}',
                interactive=True,
                elem_id="animation_prompts_main",
                info="JSON format animation prompts. Each key is a frame number, each value is the prompt for that frame."
            )
        
        # Prompt utilities
        with gr.Accordion("🛠️ Prompt Utilities", open=False):
            with FormRow():
                load_prompts_btn = gr.Button(
                    "📁 Load Prompts from File",
                    variant="secondary"
                )
                save_prompts_btn = gr.Button(
                    "💾 Save Prompts to File", 
                    variant="secondary"
                )
                validate_prompts_btn = gr.Button(
                    "✅ Validate JSON",
                    variant="primary"
                )
            
            # Validation results
            validation_output = gr.Textbox(
                label="Validation Results",
                lines=3,
                interactive=False,
                placeholder="Click 'Validate JSON' to check your prompts format..."
            )
        
        # Positive/Negative prompts (advanced)
        with gr.Accordion("➕➖ Advanced Positive/Negative Prompts", open=False):
            gr.Markdown("""
            **Advanced Mode:** Separate positive and negative prompts for fine control.
            Leave empty to use the main prompts above.
            """)
            
            with FormRow():
                animation_prompts_positive = gr.Textbox(
                    label="Positive prompts only",
                    lines=8,
                    value="",
                    interactive=True,
                    placeholder='{\n    "0": "beautiful, detailed, high quality"\n}',
                    info="Optional: Positive prompts only (overrides main prompts if provided)"
                )
            
            with FormRow():
                animation_prompts_negative = gr.Textbox(
                    label="Negative prompts only",
                    lines=8,
                    value="",
                    interactive=True,
                    placeholder='{\n    "0": "blurry, low quality, distorted"\n}',
                    info="Optional: Negative prompts for each frame"
                )
        
        # Event handlers
        def validate_prompts_handler(prompts_text):
            """Validate the JSON format of prompts."""
            try:
                import json
                
                if not prompts_text.strip():
                    return "❌ Empty prompts. Please add some prompts."
                
                # Try to parse as JSON
                prompts = json.loads(prompts_text)
                
                if not isinstance(prompts, dict):
                    return "❌ Prompts must be a JSON object (dictionary)"
                
                # Validate frame numbers
                frame_numbers = []
                for key in prompts.keys():
                    try:
                        frame_num = int(key)
                        frame_numbers.append(frame_num)
                    except ValueError:
                        return f"❌ Invalid frame number: '{key}'. Frame numbers must be integers."
                
                # Check for negative frame numbers
                if any(f < 0 for f in frame_numbers):
                    return "❌ Frame numbers cannot be negative."
                
                frame_numbers.sort()
                num_prompts = len(prompts)
                max_frame = max(frame_numbers) if frame_numbers else 0
                
                return f"""✅ **Valid JSON!** 
                
📊 **Statistics:**
• {num_prompts} prompt(s) defined
• Frame range: {min(frame_numbers)} to {max_frame}
• Total animation length: {max_frame + 1} frames

🎬 **Ready for animation generation!**"""
                
            except json.JSONDecodeError as e:
                return f"❌ **JSON Error:** {str(e)}\n\nPlease check your JSON syntax (quotes, commas, brackets)."
            except Exception as e:
                return f"❌ **Validation Error:** {str(e)}"
        
        # Connect validation handler
        if validate_prompts_btn is not None and hasattr(validate_prompts_btn, '_id'):
            valid_inputs = [comp for comp in [animation_prompts] if comp is not None and hasattr(comp, '_id')]
            valid_outputs = [comp for comp in [validation_output] if comp is not None and hasattr(comp, '_id')]
            if valid_inputs and valid_outputs:
                try:
                    validate_prompts_btn.click(
                        fn=validate_prompts_handler,
                        inputs=valid_inputs,
                        outputs=valid_outputs
                    )
                except Exception as e:
                    print(f"⚠️ Failed to connect validation button: {e}")
        
        # Auto-validate on change (debounced) - safely
        if animation_prompts is not None and hasattr(animation_prompts, '_id'):
            valid_outputs = [comp for comp in [validation_output] if comp is not None and hasattr(comp, '_id')]
            if valid_outputs:
                try:
                    animation_prompts.change(
                        fn=validate_prompts_handler,
                        inputs=[animation_prompts], 
                        outputs=valid_outputs
                    )
                except Exception as e:
                    print(f"⚠️ Failed to connect animation prompts change handler: {e}")
    
    # Return only the actual UI components created in this function
    return {
        'animation_prompts': animation_prompts,
        'load_prompts_btn': load_prompts_btn,
        'save_prompts_btn': save_prompts_btn,
        'validate_prompts_btn': validate_prompts_btn,
        'validation_output': validation_output,
        'animation_prompts_positive': animation_prompts_positive,
        'animation_prompts_negative': animation_prompts_negative,
    } 