#!/usr/bin/env python3
"""
Wan Simple Integration with Styled Progress
Updated to use experimental render core styling for progress indicators
"""

from pathlib import Path
from typing import List, Dict, Optional
import torch
import os
import numpy as np
import time
import sys
import json
from decimal import Decimal

# Import our new styled progress utilities
from .utils.wan_progress_utils import (
    WanModelLoadingContext, WanGenerationContext,
    print_wan_info, print_wan_success, print_wan_warning, print_wan_error, print_wan_progress,
    create_wan_clip_progress, create_wan_frame_progress, create_wan_inference_progress
)

# Import Deforum utilities for settings and audio handling
from ..video_audio_utilities import download_audio
from ..subtitle_handler import init_srt_file, write_frame_subtitle, calculate_frame_duration
from ..settings import save_settings_from_animation_run

class WanSimpleIntegration:
    """Simplified Wan integration with auto-discovery and proper progress styling"""
    
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.models = []
        self.pipeline = None
        self.model_size = None
        self.optimal_width = 1280  # Wan 2.2 TI2V default: 720p landscape (divisible by 32)
        self.optimal_height = 736  # Aligned to 32 for Wan 2.2 (VAE scale=16 * patch=2)
        self.flash_attention_mode = "auto"  # auto, enabled, disabled
        print_wan_info(f"Simple Integration initialized on {self.device}")
    
    def discover_models(self) -> List[Dict]:
        """Discover available Wan models with styled progress"""
        models = []
        search_paths = [
            Path("models/wan"),
            Path("models/Wan"),
            Path("models"),
            Path("../models/wan"),
            Path("../models/Wan"),
        ]
        
        print_wan_progress("Discovering Wan models...")
        
        for search_path in search_paths:
            if search_path.exists():
                print_wan_info(f"🔍 Searching: {search_path}")
                
                for model_dir in search_path.iterdir():
                    # Check for Wan 2.2, Wan 2.1, or generic wan directories
                    if model_dir.is_dir() and model_dir.name.lower().startswith(('wan2.2', 'wan2.1', 'wan-', 'wan_')):
                        model_info = self._analyze_model_directory(model_dir)
                        if model_info:
                            models.append(model_info)
                            print_wan_success(f"Found: {model_info['name']} ({model_info['type']}, {model_info['size']})")
        
        if not models:
            print_wan_warning("No Wan models found in search paths")
        else:
            print_wan_success(f"Discovery complete - found {len(models)} model(s)")
            
        return models
    
    def _analyze_model_directory(self, model_dir: Path) -> Optional[Dict]:
        """Analyze a model directory and return model info if valid"""
        if not model_dir.is_dir():
            return None
            
        # Check if this looks like a Wan model
        model_name = model_dir.name.lower()
        if 'wan' not in model_name and not any(file.name.startswith('wan') for file in model_dir.rglob('*') if file.is_file()):
            return None
        
        # Check for required model files
        if not self._has_required_files(model_dir):
            return None
        
        # Determine model type and size (Wan 2.2 first, then 2.1)
        model_type = "Unknown"
        model_size = "Unknown"

        # Detect type (prioritize Wan 2.2 models)
        if 'ti2v' in model_name:
            model_type = "TI2V"
        elif 's2v' in model_name:
            model_type = "S2V"
        elif 'animate' in model_name:
            model_type = "Animate"
        elif 't2v' in model_name:
            model_type = "T2V"
        elif 'i2v' in model_name:
            model_type = "I2V"

        # Detect size (Wan 2.2 and 2.1)
        if '5b' in model_name or '5_b' in model_name:
            model_size = "5B"
        elif 'a14b' in model_name or 'a_14b' in model_name:
            model_size = "A14B"
        elif '1.3b' in model_name:
            model_size = "1.3B"
        elif '14b' in model_name:
            model_size = "14B"

        # Detect quantization
        quantization = "FP16"  # Default
        if 'fp8' in model_name or 'e4m3fn' in model_name or 'e5m2' in model_name:
            quantization = "FP8"
        elif 'gguf' in model_name or 'q8' in model_name or 'q5' in model_name or 'q4' in model_name:
            quantization = "GGUF"
        elif 'bf16' in model_name:
            quantization = "BF16"

        return {
            'name': model_dir.name,
            'path': str(model_dir.absolute()),
            'type': model_type,
            'size': model_size,
            'quantization': quantization,
            'directory': model_dir
        }
    
    def _has_required_files(self, model_dir: Path) -> bool:
        """Check if model directory has required files"""
        required_files = [
            "config.json",
            "model_index.json"
        ]
        
        # Check for model weight files
        has_weights = any(
            file.suffix in ['.safetensors', '.bin', '.pt', '.pth']
            for file in model_dir.rglob('*')
            if file.is_file()
        )
        
        has_config = any(
            (model_dir / req_file).exists()
            for req_file in required_files
        )
        
        return has_weights and has_config
    
    
    def get_best_model(self) -> Optional[Dict]:
        """Get the best available model"""
        if not self.models:
            self.discover_models()
        
        if not self.models:
            return None
        
        # Priority: TI2V > T2V > I2V (Wan 2.2 preferred), 5B > 1.3B > 14B > A14B, FP8 > GGUF > FP16 (VRAM efficient)
        def model_priority(model):
            type_priority = {'TI2V': 0, 'T2V': 1, 'I2V': 2, 'S2V': 3, 'Animate': 4, 'Unknown': 5}
            size_priority = {'5B': 0, '1.3B': 1, '14B': 2, 'A14B': 3, 'Unknown': 4}
            quant_priority = {'FP8': 0, 'GGUF': 1, 'FP16': 2, 'BF16': 3}  # FP8 preferred for VRAM efficiency
            return (
                type_priority.get(model['type'], 5),
                size_priority.get(model['size'], 4),
                quant_priority.get(model.get('quantization', 'FP16'), 2)
            )

        best_model = min(self.models, key=model_priority)
        quantization_info = best_model.get('quantization', 'Unknown')
        print(f"🎯 Best model selected: {best_model['name']} ({best_model['type']}, {best_model['size']}, {quantization_info})")
        return best_model
    
    def load_simple_wan_pipeline(self, model_info: Dict, wan_args=None) -> bool:
        """Load Wan pipeline with styled progress indicators"""
        model_name = model_info['name']

        with WanModelLoadingContext(model_name) as progress:
            try:
                progress.update(10, "Initializing...")
                progress.update(30, "Loading model...")
                success = self._load_standard_wan_model(model_info)

                if success:
                    progress.update(80, "Configuring...")
                    # Configure Flash Attention if requested
                    if wan_args and hasattr(wan_args, 'wan_flash_attention'):
                        self.flash_attention_mode = wan_args.wan_flash_attention
                        print_wan_info(f"Flash Attention mode: {self.flash_attention_mode}")

                    progress.update(100, "Complete!")
                    return True
                else:
                    print_wan_error(f"Failed to load pipeline for {model_name}")
                    return False

            except Exception as e:
                print_wan_error(f"Model loading failed: {e}")
                return False
    
    def _load_standard_wan_model(self, model_info: Dict) -> bool:
        """Load standard T2V/I2V Wan model"""
        try:
            # Strategy 1: Try official Wan implementation
            extension_root = Path(__file__).parent.parent.parent.parent
            wan_repo_path = extension_root / "Wan2.1"
            
            if wan_repo_path.exists() and (wan_repo_path / "wan").exists():
                print(f"🔧 Trying official Wan implementation from: {wan_repo_path}")
                
                if str(wan_repo_path) not in sys.path:
                    sys.path.insert(0, str(wan_repo_path))
                
                try:
                    import wan
                    from wan.text2video import WanT2V
                    
                    # Apply Flash Attention patches AFTER Wan modules are imported
                    try:
                        from .wan_flash_attention_patch import apply_flash_attention_patch, update_patched_flash_attention_mode
                        
                        # Update mode if stored
                        if hasattr(self, 'flash_attention_mode'):
                            update_patched_flash_attention_mode(self.flash_attention_mode)
                        
                        success = apply_flash_attention_patch()
                        if success:
                            print("✅ Flash Attention monkey patch applied successfully")
                        else:
                            print("⚠️ Flash Attention patch could not be applied - may be already patched")
                    except Exception as patch_e:
                        print(f"⚠️ Flash Attention patch failed: {patch_e}")
                        print("🔄 Continuing without patches...")
                    
                    print("🚀 Loading with official Wan T2V...")
                    
                    # Create minimal config
                    class MinimalConfig:
                        def __init__(self):
                            self.model = type('obj', (object,), {
                                'num_attention_heads': 32,
                                'attention_head_dim': 128,
                                'in_channels': 4,
                                'out_channels': 4,
                                'num_layers': 28,
                                'sample_size': 32,
                                'patch_size': 2,
                            })
                    
                    config = MinimalConfig()
                    
                    t2v_model = WanT2V(
                        config=config,
                        checkpoint_dir=model_info['path'],
                        device_id=0,
                        rank=0,
                        dit_fsdp=False,
                        t5_fsdp=False
                    )
                    
                    # Create wrapper
                    class WanWrapper:
                        def __init__(self, t2v_model):
                            self.t2v_model = t2v_model
                        
                        def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                            # Ensure dimensions are aligned
                            aligned_width = ((width + 15) // 16) * 16
                            aligned_height = ((height + 15) // 16) * 16
                            
                            if aligned_width != width or aligned_height != height:
                                print(f"🔧 Dimension alignment: {width}x{height} -> {aligned_width}x{aligned_height}")
                            
                            return self.t2v_model.generate(
                                input_prompt=prompt,
                                size=(aligned_width, aligned_height),
                                frame_num=num_frames,
                                sampling_steps=num_inference_steps,
                                guide_scale=guidance_scale,
                                shift=5.0,
                                sample_solver='unipc',
                                offload_model=True,
                                **kwargs
                            )
                        
                        def generate_image2video(self, image, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                            # For I2V, enhance prompt for continuity
                            enhanced_prompt = f"Starting from the given image, {prompt}. Maintain visual continuity."
                            return self.__call__(enhanced_prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs)
                    
                    self.pipeline = WanWrapper(t2v_model)
                    print("✅ Official Wan model loaded successfully")
                    return True
                    
                except Exception as wan_e:
                    print(f"⚠️ Official Wan loading failed: {wan_e}")
                    print("🔄 Trying diffusers fallback...")
            
            # Strategy 2: Try diffusers fallback
            try:
                # Check if this is a Wan 2.2 Diffusers model
                is_wan22_diffusers = (
                    'TI2V' in model_info['type'] or
                    '5B' in model_info['size'] or
                    'A14B' in model_info['size'] or
                    'wan2.2' in model_info['name'].lower()
                )

                # Initialize flags for memory optimization
                use_aggressive_offload = False

                if is_wan22_diffusers:
                    print("🔄 Loading Wan 2.2 Diffusers pipeline...")

                    # Apply compatibility patches BEFORE importing diffusers
                    # This ensures patches are active even if diffusers was imported elsewhere
                    try:
                        from ..diffusers_compat_patch import apply_all_patches
                        print("🔧 Applying diffusers compatibility patches before pipeline load...")
                        apply_all_patches()
                    except Exception as patch_e:
                        print(f"⚠️ Warning: Compatibility patches failed: {patch_e}")
                        import traceback
                        traceback.print_exc()

                    from diffusers import WanPipeline, AutoencoderKLWan

                    # Enable aggressive offload for TI2V-5B models (16GB VRAM optimization)
                    # A14B models are too large and need >24GB VRAM anyway
                    is_5b_model = '5B' in model_info['size'] or '5b' in model_info['name'].lower()

                    if is_5b_model:
                        print_wan_info("🔍 TI2V-5B model detected - enabling aggressive VRAM optimizations for 16GB GPUs")
                        model_dtype = torch.float16
                        vae_dtype = torch.float16  # FP16 VAE for lower VRAM
                        use_aggressive_offload = True
                    else:
                        print_wan_info("🔍 Large model detected (A14B) - standard precision")
                        model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                        vae_dtype = torch.float32  # Standard VAE needs float32 for stability
                        use_aggressive_offload = False

                    # Load VAE separately for better compatibility (keep on CPU initially)
                    print_wan_info(f"Loading VAE with dtype={vae_dtype}")
                    vae = AutoencoderKLWan.from_pretrained(
                        model_info['path'],
                        subfolder="vae",
                        torch_dtype=vae_dtype,
                        low_cpu_mem_usage=True  # Reduce CPU RAM usage during load
                    )

                    # Load main pipeline (keep on CPU initially)
                    print_wan_info(f"Loading pipeline with dtype={model_dtype}")
                    pipeline = WanPipeline.from_pretrained(
                        model_info['path'],
                        vae=vae,
                        torch_dtype=model_dtype,
                        low_cpu_mem_usage=True,  # Reduce CPU RAM usage during load
                        use_safetensors=True
                    )

                    if is_5b_model:
                        print_wan_success("✅ Loaded Wan 2.2 TI2V-5B pipeline with VRAM optimizations")
                    else:
                        print_wan_success("✅ Loaded Wan 2.2 pipeline")
                else:
                    # Fallback to generic DiffusionPipeline
                    print("🔄 Loading with generic DiffusionPipeline...")
                    from diffusers import DiffusionPipeline

                    pipeline = DiffusionPipeline.from_pretrained(
                        model_info['path'],
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        use_safetensors=True
                    )

                # Enable memory optimizations based on model type
                if torch.cuda.is_available():
                    if is_wan22_diffusers and use_aggressive_offload:
                        print_wan_info("🔧 Enabling aggressive VRAM optimizations for 16GB VRAM...")

                        # Sequential CPU offload (more aggressive than model CPU offload)
                        try:
                            pipeline.enable_sequential_cpu_offload()
                            print_wan_success("✅ Sequential CPU offload enabled")
                        except:
                            # Fallback to model CPU offload
                            pipeline.enable_model_cpu_offload()
                            print_wan_success("✅ Model CPU offload enabled (fallback)")

                        # Enable attention slicing for lower VRAM
                        try:
                            pipeline.enable_attention_slicing(slice_size="auto")
                            print_wan_success("✅ Attention slicing enabled")
                        except Exception as e:
                            print_wan_warning(f"Attention slicing not available: {e}")

                        # Enable VAE tiling for lower VRAM
                        try:
                            if hasattr(pipeline, 'enable_vae_tiling'):
                                pipeline.enable_vae_tiling()
                                print_wan_success("✅ VAE tiling enabled")
                        except Exception as e:
                            print_wan_warning(f"VAE tiling not available: {e}")

                        # Enable VAE slicing
                        try:
                            if hasattr(pipeline, 'enable_vae_slicing'):
                                pipeline.enable_vae_slicing()
                                print_wan_success("✅ VAE slicing enabled")
                        except Exception as e:
                            print_wan_warning(f"VAE slicing not available: {e}")

                        print_wan_success("✅ All VRAM optimizations enabled - should work with <16GB VRAM")

                    else:
                        # Standard CPU offload for non-FP8 models
                        print_wan_info("🔧 Enabling model CPU offload to save VRAM...")
                        pipeline.enable_model_cpu_offload()
                        print_wan_success("✅ CPU offload enabled - model will stream to GPU during inference")
                
                # Create wrapper for diffusers pipeline
                class DiffusersWrapper:
                    def __init__(self, pipeline):
                        self.pipeline = pipeline
                    
                    def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                        # Wan 2.2 requires dimensions divisible by 32 (VAE spatial_scale=16 * transformer patch_size=2)
                        aligned_width = ((width + 31) // 32) * 32
                        aligned_height = ((height + 31) // 32) * 32

                        # WanPipeline requires specific parameter names
                        import inspect
                        pipeline_signature = inspect.signature(self.pipeline.__call__)

                        print_wan_info(f"🔍 Pipeline parameters available: {list(pipeline_signature.parameters.keys())}")

                        generation_kwargs = {
                            "prompt": prompt,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                        }

                        # Add video-specific parameters based on pipeline signature
                        if 'height' in pipeline_signature.parameters:
                            generation_kwargs['height'] = aligned_height
                        if 'width' in pipeline_signature.parameters:
                            generation_kwargs['width'] = aligned_width
                        if 'num_frames' in pipeline_signature.parameters:
                            generation_kwargs['num_frames'] = num_frames
                        elif 'video_length' in pipeline_signature.parameters:
                            generation_kwargs['video_length'] = num_frames
                        elif 'num_video_frames' in pipeline_signature.parameters:
                            generation_kwargs['num_video_frames'] = num_frames

                        print_wan_info(f"🎬 Generation parameters:")
                        print_wan_info(f"   Resolution: {aligned_width}x{aligned_height}")
                        print_wan_info(f"   Frames: {num_frames}")
                        print_wan_info(f"   Steps: {num_inference_steps}")
                        print_wan_info(f"   Guidance: {guidance_scale}")
                        print_wan_info(f"   Full kwargs: {generation_kwargs}")

                        with torch.no_grad():
                            return self.pipeline(**generation_kwargs)
                    
                    def generate_image2video(self, image, prompt, height, width, num_frames, num_inference_steps, guidance_scale, strength=0.8, **kwargs):
                        """
                        Image-to-Video generation with proper image conditioning

                        Args:
                            image: PIL Image to use as first frame/conditioning
                            prompt: Text prompt for video generation
                            strength: Image conditioning strength (0.0-1.0)
                                     1.0 = maximum continuity from image
                                     0.0 = ignore image, pure T2V
                        """
                        # Wan 2.2 requires dimensions divisible by 32
                        aligned_width = ((width + 31) // 32) * 32
                        aligned_height = ((height + 31) // 32) * 32

                        # Enhanced prompt for I2V continuity
                        # More specific language to encourage smooth transitions
                        enhanced_prompt = f"{prompt}. Smooth continuation, maintaining consistent style and subject."

                        import inspect
                        pipeline_signature = inspect.signature(self.pipeline.__call__)

                        generation_kwargs = {
                            "prompt": enhanced_prompt,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                        }

                        # Add image conditioning if pipeline supports it
                        if 'image' in pipeline_signature.parameters:
                            generation_kwargs['image'] = image
                            print_wan_info(f"✅ I2V conditioning: Using image with strength {strength:.2f}")
                        else:
                            print_wan_warning("⚠️ Pipeline does not support image parameter - using enhanced prompt only")

                        # Add strength parameter if supported (some I2V pipelines use this)
                        if 'strength' in pipeline_signature.parameters:
                            generation_kwargs['strength'] = strength

                        # Add image_guidance_scale if supported (alternative to strength in some pipelines)
                        if 'image_guidance_scale' in pipeline_signature.parameters:
                            # Convert strength to guidance scale (typically 1.0-10.0 range)
                            image_guidance = 1.0 + (strength * 9.0)  # Map 0-1 to 1-10
                            generation_kwargs['image_guidance_scale'] = image_guidance
                            print_wan_info(f"🎯 Image guidance scale: {image_guidance:.1f}")

                        # Add resolution parameters
                        if 'height' in pipeline_signature.parameters:
                            generation_kwargs['height'] = aligned_height
                        if 'width' in pipeline_signature.parameters:
                            generation_kwargs['width'] = aligned_width
                        if 'num_frames' in pipeline_signature.parameters:
                            generation_kwargs['num_frames'] = num_frames
                        elif 'video_length' in pipeline_signature.parameters:
                            generation_kwargs['video_length'] = num_frames
                        elif 'num_video_frames' in pipeline_signature.parameters:
                            generation_kwargs['num_video_frames'] = num_frames

                        print_wan_info(f"🎬 I2V Generation:")
                        print_wan_info(f"   Resolution: {aligned_width}x{aligned_height}")
                        print_wan_info(f"   Frames: {num_frames}")
                        print_wan_info(f"   I2V Strength: {strength:.2f}")
                        print_wan_info(f"   CFG: {guidance_scale}")

                        with torch.no_grad():
                            return self.pipeline(**generation_kwargs)
                
                self.pipeline = DiffusersWrapper(pipeline)
                print("✅ Diffusers model loaded successfully")
                return True
                
            except Exception as diffusers_e:
                print(f"❌ Diffusers loading failed: {diffusers_e}")
                
                raise RuntimeError(f"""
❌ CRITICAL: Could not load Wan 2.2 model!

🔧 TROUBLESHOOTING:
1. 📦 Dependencies upgraded automatically by extension
2. 💾 Verify model download is complete: {model_info['path']}
3. 🔄 Restart WebUI to apply dependency upgrades
4. 💬 Check console for detailed error messages

❌ Model loading failed!
Error: {diffusers_e}
""")
        
        except Exception as e:
            print(f"❌ Standard model loading failed: {e}")
            return False
    
    def generate_video_with_i2v_chaining(self, clips, model_info, output_dir, wan_args=None, **kwargs):
        """Generate video with I2V chaining using styled progress indicators"""
        if not self.pipeline:
            raise RuntimeError("Pipeline not loaded")
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestring for consistent file naming
            timestring = kwargs.pop('timestring', str(int(time.time())))

            # Handle audio download early if provided
            audio_url = kwargs.get('soundtrack_path') or kwargs.get('audio_url')
            cached_audio_path = None
            if audio_url:
                cached_audio_path = self.download_and_cache_audio(audio_url, output_dir, timestring)
                kwargs['cached_audio_path'] = cached_audio_path

            # Save settings file before generation (timestring removed from kwargs to avoid duplicate)
            settings_file = self.save_wan_settings_and_metadata(
                output_dir, timestring, clips, model_info, wan_args, **kwargs
            )
            
            # Create SRT subtitle file
            fps = kwargs.get('fps', 8.0)
            srt_file = self.create_wan_srt_file(output_dir, timestring, clips, fps)
            
            with WanGenerationContext(len(clips)) as gen_context:
                all_frame_paths = []
                last_frame_path = None
                total_frame_idx = 0
                
                # Get generation parameters
                steps = kwargs.get('num_inference_steps', 50)
                guidance_scale = kwargs.get('guidance_scale', 7.5)
                height = kwargs.get('height', self.optimal_height)
                width = kwargs.get('width', self.optimal_width)
                
                print_wan_info(f"Model: {model_info['name']} ({model_info['type']}, {model_info['size']})")
                print_wan_info(f"Output: {output_dir}")
                if settings_file:
                    print_wan_info(f"Settings: {os.path.basename(settings_file)}")
                if srt_file:
                    print_wan_info(f"Subtitles: {os.path.basename(srt_file)}")
                if cached_audio_path:
                    print_wan_info(f"Audio: {os.path.basename(cached_audio_path)}")
                
                # Generate each clip with progress tracking
                for clip_idx, clip in enumerate(clips):
                    try:
                        # Create frame progress bar for this clip
                        with create_wan_frame_progress(clip['num_frames'], clip_idx) as frame_progress:
                            gen_context.update_clip(clip_idx, clip['prompt'][:30])
                            
                            print_wan_progress(f"Generating clip {clip_idx + 1}/{len(clips)}")
                            print_wan_info(f"Prompt: {clip['prompt'][:50]}...")
                            print_wan_info(f"Frames: {clip['num_frames']}")
                            
                            # Create inference progress bar
                            with create_wan_inference_progress(steps) as inference_progress:
                                # Generate based on chaining mode
                                if clip_idx == 0 or last_frame_path is None:
                                    # First clip: T2V generation
                                    print_wan_info("Using T2V generation for first clip")
                                    result = self.pipeline(
                                        prompt=clip['prompt'],
                                        height=height,
                                        width=width,
                                        num_frames=clip['num_frames'],
                                        num_inference_steps=steps,
                                        guidance_scale=guidance_scale,
                                    )
                                    inference_progress.update(steps)
                                else:
                                    # Subsequent clips: I2V chaining
                                    print_wan_info(f"I2V chaining from: {os.path.basename(last_frame_path)}")

                                    # Get I2V strength from wan_args
                                    i2v_strength = 0.85  # Default: strong continuity (matches UI default)
                                    if wan_args and hasattr(wan_args, 'wan_strength_override') and wan_args.wan_strength_override:
                                        if hasattr(wan_args, 'wan_fixed_strength'):
                                            i2v_strength = float(wan_args.wan_fixed_strength)
                                            print_wan_info(f"Using fixed I2V strength from settings: {i2v_strength:.2f}")
                                    else:
                                        print_wan_info(f"Using default I2V strength: {i2v_strength:.2f}")

                                    if hasattr(self.pipeline, 'generate_image2video'):
                                        # Custom I2V pipeline with image conditioning
                                        from PIL import Image
                                        last_frame_image = Image.open(last_frame_path)

                                        result = self.pipeline.generate_image2video(
                                            image=last_frame_image,
                                            prompt=clip['prompt'],
                                            height=height,
                                            width=width,
                                            num_frames=clip['num_frames'],
                                            num_inference_steps=steps,
                                            guidance_scale=guidance_scale,
                                            strength=i2v_strength,  # Control image conditioning strength
                                        )
                                        inference_progress.update(steps)
                                    else:
                                        # Fallback to T2V with enhanced prompt
                                        print_wan_warning("⚠️ Pipeline does not support I2V - using enhanced T2V")
                                        enhanced_prompt = f"Continuing seamlessly from previous scene, {clip['prompt']}. Maintain visual continuity, smooth transition."
                                        result = self.pipeline(
                                            prompt=enhanced_prompt,
                                            height=height,
                                            width=width,
                                            num_frames=clip['num_frames'],
                                            num_inference_steps=steps,
                                            guidance_scale=guidance_scale,
                                        )
                                        inference_progress.update(steps)
                            
                            # Process and save frames with progress
                            clip_frames = self._process_and_save_frames(
                                result, clip_idx, output_dir, timestring, 
                                total_frame_idx, frame_progress
                            )
                            
                            if clip_frames:
                                all_frame_paths.extend(clip_frames)
                                total_frame_idx += len(clip_frames)
                                last_frame_path = clip_frames[-1]  # Update for next clip
                                print_wan_success(f"Generated {len(clip_frames)} frames for clip {clip_idx + 1}")
                            else:
                                raise RuntimeError(f"No frames generated for clip {clip_idx + 1}")
                    
                    except Exception as e:
                        print_wan_error(f"Clip {clip_idx + 1} generation failed: {e}")
                        raise
                
                print_wan_success(f"All clips generated! Total frames: {len(all_frame_paths)}")
                print_wan_info(f"Frames saved to: {output_dir}")
                
                # Return comprehensive results
                return {
                    'output_dir': output_dir,
                    'frame_paths': all_frame_paths,
                    'settings_file': settings_file,
                    'srt_file': srt_file,
                    'audio_file': cached_audio_path,
                    'timestring': timestring,
                    'total_frames': len(all_frame_paths),
                    'total_clips': len(clips)
                }
                
        except Exception as e:
            print_wan_error(f"I2V chained video generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"I2V chained video generation failed: {e}")
        finally:
            # Clean up if needed
            pass
    
    def _process_and_save_frames(self, result, clip_idx, output_dir, timestring, start_frame_idx, frame_progress=None):
        """Process generation result and save frames with progress tracking"""
        try:
            from PIL import Image
            import numpy as np

            frames = []

            # Debug: Log result type
            print_wan_info(f"🔍 Result type: {type(result)}")
            print_wan_info(f"🔍 Result attributes: {dir(result)[:10]}...")  # First 10 attributes

            # Handle different result formats
            if isinstance(result, tuple):
                print_wan_info("Result is tuple, extracting first element")
                frames_data = result[0]
            elif hasattr(result, 'frames'):
                print_wan_info("Result has .frames attribute")
                frames_data = result.frames
            elif hasattr(result, 'images'):
                print_wan_info("Result has .images attribute")
                frames_data = result.images
            elif hasattr(result, 'videos'):
                print_wan_info("Result has .videos attribute")
                frames_data = result.videos
            else:
                print_wan_info("Using result directly as frames_data")
                frames_data = result

            print_wan_info(f"🔍 Frames data type: {type(frames_data)}")

            # Check if frames_data is None or empty
            if frames_data is None:
                print_wan_error("❌ Frames data is None!")
                return []

            # Convert to individual frames
            if hasattr(frames_data, 'cpu'):  # Tensor
                frames_tensor = frames_data.cpu()
                print_wan_info(f"✅ Processing tensor: {frames_tensor.shape}")
                
                # Handle different tensor formats
                if len(frames_tensor.shape) == 5:  # (B, C, F, H, W)
                    frames_tensor = frames_tensor.squeeze(0)  # Remove batch dimension
                
                if len(frames_tensor.shape) == 4:  # (C, F, H, W)
                    for frame_idx in range(frames_tensor.shape[1]):
                        frame_tensor = frames_tensor[:, frame_idx, :, :]  # (C, H, W)
                        frame_np = frame_tensor.permute(1, 2, 0).numpy()  # (H, W, C)
                        
                        # Models may output in [-1, 1] range, convert to [0, 255]
                        if frame_np.min() < -0.5:  # Likely [-1, 1] range
                            # Convert from [-1, 1] to [0, 1] then to [0, 255]
                            frame_np = (frame_np + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                            frame_np = np.clip(frame_np, 0, 1)  # Ensure valid range
                            frame_np = (frame_np * 255).astype(np.uint8)
                        elif frame_np.max() <= 1.0:  # [0, 1] range
                            frame_np = (frame_np * 255).astype(np.uint8)
                        else:  # Already in [0, 255] range
                            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                        
                        frames.append(frame_np)
                        
                        # Update progress if available
                        if frame_progress:
                            frame_progress.update(1)
                else:
                    print_wan_warning(f"Unexpected tensor shape: {frames_tensor.shape}")
                    # Try to treat as single frame
                    if len(frames_tensor.shape) == 3:  # (C, H, W)
                        frame_np = frames_tensor.permute(1, 2, 0).numpy()
                        
                        # Apply same normalization fix
                        if frame_np.min() < -0.5:  # Likely [-1, 1] range
                            frame_np = (frame_np + 1.0) / 2.0
                            frame_np = np.clip(frame_np, 0, 1)
                            frame_np = (frame_np * 255).astype(np.uint8)
                        elif frame_np.max() <= 1.0:
                            frame_np = (frame_np * 255).astype(np.uint8)
                        else:
                            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                            
                        frames.append(frame_np)
                        if frame_progress:
                            frame_progress.update(1)
            
            elif isinstance(frames_data, list):  # List of PIL Images or arrays
                print_wan_info(f"✅ Processing list with {len(frames_data)} items")
                for frame in frames_data:
                    if hasattr(frame, 'save'):  # PIL Image
                        frames.append(np.array(frame))
                    else:
                        frames.append(frame)

                    if frame_progress:
                        frame_progress.update(1)

            elif isinstance(frames_data, np.ndarray):  # Numpy array (WanPipelineOutput.frames)
                print_wan_info(f"✅ Processing numpy array: {frames_data.shape}")

                # WanPipeline returns frames in shape (B, F, H, W, C) or (B, C, F, H, W)
                if len(frames_data.shape) == 5:
                    # Check if channels are first (B, C, F, H, W) or last (B, F, H, W, C)
                    if frames_data.shape[1] == 3 or frames_data.shape[1] == 4:
                        # (B, C, F, H, W) format
                        print_wan_info("Detected (B, C, F, H, W) format, converting...")
                        frames_np = frames_data[0]  # Remove batch dim: (C, F, H, W)
                        num_frames_in_clip = frames_np.shape[1]

                        for frame_idx in range(num_frames_in_clip):
                            frame = frames_np[:, frame_idx, :, :]  # (C, H, W)
                            frame = np.transpose(frame, (1, 2, 0))  # (H, W, C)

                            # Normalize to [0, 255]
                            if frame.min() < -0.5:  # [-1, 1] range
                                frame = (frame + 1.0) / 2.0
                                frame = np.clip(frame, 0, 1)
                                frame = (frame * 255).astype(np.uint8)
                            elif frame.max() <= 1.0:  # [0, 1] range
                                frame = (frame * 255).astype(np.uint8)
                            else:  # Already [0, 255]
                                frame = np.clip(frame, 0, 255).astype(np.uint8)

                            frames.append(frame)
                            if frame_progress:
                                frame_progress.update(1)
                    else:
                        # (B, F, H, W, C) format
                        print_wan_info("Detected (B, F, H, W, C) format, converting...")
                        frames_np = frames_data[0]  # Remove batch dim: (F, H, W, C)

                        for frame_idx in range(frames_np.shape[0]):
                            frame = frames_np[frame_idx]  # (H, W, C)

                            # Normalize to [0, 255]
                            if frame.min() < -0.5:  # [-1, 1] range
                                frame = (frame + 1.0) / 2.0
                                frame = np.clip(frame, 0, 1)
                                frame = (frame * 255).astype(np.uint8)
                            elif frame.max() <= 1.0:  # [0, 1] range
                                frame = (frame * 255).astype(np.uint8)
                            else:  # Already [0, 255]
                                frame = np.clip(frame, 0, 255).astype(np.uint8)

                            frames.append(frame)
                            if frame_progress:
                                frame_progress.update(1)

                elif len(frames_data.shape) == 4:
                    # (F, H, W, C) format (already removed batch)
                    print_wan_info("Detected (F, H, W, C) format, converting...")
                    for frame_idx in range(frames_data.shape[0]):
                        frame = frames_data[frame_idx]  # (H, W, C)

                        # Normalize to [0, 255]
                        if frame.min() < -0.5:
                            frame = (frame + 1.0) / 2.0
                            frame = np.clip(frame, 0, 1)
                            frame = (frame * 255).astype(np.uint8)
                        elif frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)

                        frames.append(frame)
                        if frame_progress:
                            frame_progress.update(1)
                else:
                    print_wan_error(f"❌ Unexpected numpy array shape: {frames_data.shape}")
                    return []

            else:
                # Unknown format - try to debug it
                print_wan_error(f"❌ Unknown frames_data format!")
                print_wan_error(f"   Type: {type(frames_data)}")
                print_wan_error(f"   Has __len__: {hasattr(frames_data, '__len__')}")
                print_wan_error(f"   Has __iter__: {hasattr(frames_data, '__iter__')}")
                if hasattr(frames_data, '__len__'):
                    try:
                        print_wan_error(f"   Length: {len(frames_data)}")
                    except:
                        pass
                return []

            # Check if we extracted any frames
            if not frames:
                print_wan_error(f"❌ No frames extracted from result!")
                print_wan_error(f"   Result type: {type(result)}")
                print_wan_error(f"   Frames data type: {type(frames_data)}")
                return []

            print_wan_info(f"✅ Extracted {len(frames)} frames, proceeding to save...")

            # Save frames as PNG files
            saved_paths = []
            for i, frame_np in enumerate(frames):
                frame_filename = f"{timestring}_{start_frame_idx + i:09d}.png"
                frame_path = os.path.join(output_dir, frame_filename)
                
                try:
                    pil_image = Image.fromarray(frame_np)
                    pil_image.save(frame_path)
                    saved_paths.append(frame_path)
                except Exception as save_e:
                    print_wan_warning(f"Failed to save frame {i}: {save_e}")
                    continue
            
            print_wan_success(f"Saved {len(saved_paths)} frames for clip {clip_idx + 1}")
            return saved_paths
            
        except Exception as e:
            print_wan_error(f"Frame processing failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def test_wan_setup(self) -> bool:
        """Test if Wan setup is working with styled output"""
        try:
            print_wan_progress("Testing Wan setup...")
            
            # Test model discovery
            models = self.discover_models()
            if not models:
                print_wan_error("No models found")
                return False
            
            # Test best model selection
            best_model = self.get_best_model()
            if not best_model:
                print_wan_error("No suitable model found")
                return False
            
            print_wan_success(f"Setup test passed - found {len(models)} models")
            return True
            
        except Exception as e:
            print_wan_error(f"Setup test failed: {e}")
            return False
    
    def unload_model(self):
        """Unload the model to free memory"""
        if self.pipeline:
            try:
                if hasattr(self.pipeline, 'to'):
                    self.pipeline.to('cpu')
                del self.pipeline
                self.pipeline = None
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                print_wan_success("Model unloaded, memory freed")
            except Exception as e:
                print_wan_warning(f"Error unloading model: {e}")
    
    def save_wan_settings_and_metadata(self, output_dir: str, timestring: str, clips: List[Dict], 
                                       model_info: Dict, wan_args=None, **kwargs) -> str:
        """Save Wan generation settings to match normal Deforum format"""
        try:
            settings_filename = os.path.join(output_dir, f"{timestring}_settings.txt")
            
            # Create comprehensive settings dictionary
            settings = {
                # Wan-specific settings
                "wan_model_name": model_info['name'],
                "wan_model_type": model_info['type'],
                "wan_model_size": model_info['size'],
                "wan_model_path": model_info['path'],
                "wan_flash_attention_mode": self.flash_attention_mode,
                
                # Generation parameters
                "width": kwargs.get('width', self.optimal_width),
                "height": kwargs.get('height', self.optimal_height),
                "num_inference_steps": kwargs.get('num_inference_steps', 50),
                "guidance_scale": kwargs.get('guidance_scale', 7.5),
                "fps": kwargs.get('fps', 8),
                
                # Clip information
                "total_clips": len(clips),
                "total_frames": sum(clip['num_frames'] for clip in clips),
                "clips": clips,
                
                # Metadata
                "generation_mode": "wan_i2v_chaining",
                "timestring": timestring,
                "output_directory": output_dir,
                "generation_timestamp": time.time(),
                "device": str(self.device),
                
                # Version info
                "wan_integration_version": "1.0.0",
                "deforum_git_commit_id": self._get_deforum_version(),
            }
            
            # Add wan_args if provided
            if wan_args:
                settings.update({f"wan_{k}": v for k, v in vars(wan_args).items()})
            
            # Save settings file
            with open(settings_filename, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
            
            print_wan_success(f"Settings saved: {os.path.basename(settings_filename)}")
            return settings_filename
            
        except Exception as e:
            print_wan_error(f"Failed to save settings: {e}")
            return None
    
    def create_wan_srt_file(self, output_dir: str, timestring: str, clips: List[Dict], 
                           fps: float = 8.0) -> str:
        """Create SRT subtitle file for Wan generation"""
        try:
            srt_filename = os.path.join(output_dir, f"{timestring}.srt")
            
            # Initialize SRT file
            frame_duration = init_srt_file(srt_filename, fps)
            
            # Write subtitle entries for each clip
            current_frame = 0
            for clip_idx, clip in enumerate(clips):
                clip_frames = clip['num_frames']
                
                # Create subtitle text for this clip
                subtitle_text = f"Clip {clip_idx + 1}/{len(clips)}: {clip['prompt'][:80]}"
                if len(clip['prompt']) > 80:
                    subtitle_text += "..."
                
                # Calculate timing
                start_time = Decimal(current_frame) * frame_duration
                end_time = Decimal(current_frame + clip_frames) * frame_duration
                
                # Write subtitle entry
                with open(srt_filename, "a", encoding="utf-8") as f:
                    f.write(f"{clip_idx + 1}\n")
                    f.write(f"{self._time_to_srt_format(start_time)} --> {self._time_to_srt_format(end_time)}\n")
                    f.write(f"{subtitle_text}\n\n")
                
                current_frame += clip_frames
            
            print_wan_success(f"SRT file created: {os.path.basename(srt_filename)}")
            return srt_filename
            
        except Exception as e:
            print_wan_error(f"Failed to create SRT file: {e}")
            return None
    
    def download_and_cache_audio(self, audio_url: str, output_dir: str, timestring: str) -> str:
        """Download and cache audio file in output directory"""
        try:
            if not audio_url or not audio_url.startswith(('http://', 'https://')):
                return audio_url  # Return as-is if not a URL
            
            print_wan_progress(f"Downloading audio: {audio_url}")
            
            # Download audio using Deforum's utility
            temp_audio_path = download_audio(audio_url)
            
            # Determine file extension
            _, ext = os.path.splitext(audio_url)
            if not ext:
                ext = '.mp3'  # Default to MP3
            
            # Create cached audio path in output directory
            cached_audio_path = os.path.join(output_dir, f"{timestring}_soundtrack{ext}")
            
            # Copy to output directory
            import shutil
            shutil.copy2(temp_audio_path, cached_audio_path)
            
            # Clean up temp file
            try:
                os.unlink(temp_audio_path)
            except:
                pass
            
            print_wan_success(f"Audio cached: {os.path.basename(cached_audio_path)}")
            return cached_audio_path
            
        except Exception as e:
            print_wan_error(f"Failed to download/cache audio: {e}")
            return audio_url  # Return original URL as fallback
    
    def _time_to_srt_format(self, seconds: Decimal) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours, remainder = divmod(float(seconds), 3600)
        minutes, remainder = divmod(remainder, 60)
        seconds_int, milliseconds = divmod(remainder, 1)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds_int):02},{int(milliseconds * 1000):03}"
    
    def _get_deforum_version(self) -> str:
        """Get Deforum version/commit ID"""
        try:
            from ..settings import get_deforum_version
            return get_deforum_version()
        except:
            return "unknown"

# Global instance for easy access
wan_integration = WanSimpleIntegration()

def wan_generate_video_main(*args, **kwargs):
    """Main entry point for Wan video generation with styled output"""
    try:
        print_wan_progress("Generate video main called")
        
        # Test setup first
        if not wan_integration.test_wan_setup():
            return "❌ Wan setup test failed - check model installation"
        
        # Get best model
        model_info = wan_integration.get_best_model()
        if not model_info:
            return "❌ No suitable Wan models found"
        
        # For now, return a simple success message
        return f"✅ Wan setup verified - ready to generate with {model_info['name']}"
        
    except Exception as e:
        print_wan_error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Wan generation error: {e}"

if __name__ == "__main__":
    # Test the integration
    integration = WanSimpleIntegration()
    integration.test_wan_setup() 