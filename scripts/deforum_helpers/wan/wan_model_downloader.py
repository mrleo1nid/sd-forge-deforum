#!/usr/bin/env python3
"""
Wan Model Downloader
Automatically downloads Wan models from HuggingFace when needed
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, List
import time

class WanModelDownloader:
    """Handles automatic downloading of Wan models"""
    
    def __init__(self):
        # Dynamically detect the models directory
        self.models_dir = self._detect_models_directory()
        self.available_models = {
            # Kijai's FP8 Quantized Models (Recommended for <16GB VRAM)
            "TI2V-5B-FP8": {
                "repo_id": "Kijai/WanVideo_comfy_fp8_scaled",
                "local_dir": str(self.models_dir / "Wan2.2-TI2V-5B-FP8-Kijai"),
                "description": "Wan 2.2 TI2V-5B FP8 (Unified T2V + I2V, 720P@24fps, <16GB VRAM) [Kijai]",
                "size_gb": 12,
                "quantization": "FP8",
                "vram": "~12GB",
                "source": "Kijai"
            },
            "TI2V-5B-GGUF": {
                "repo_id": "Kijai/WanVideo_comfy_GGUF",
                "local_dir": str(self.models_dir / "Wan2.2-TI2V-5B-GGUF-Kijai"),
                "description": "Wan 2.2 TI2V-5B GGUF (For very low VRAM, experimental) [Kijai]",
                "size_gb": 8,
                "quantization": "GGUF",
                "vram": "~8GB",
                "source": "Kijai"
            },
            "T2V-A14B-FP8": {
                "repo_id": "Kijai/WanVideo_comfy_fp8_scaled",
                "local_dir": str(self.models_dir / "Wan2.2-T2V-A14B-FP8-Kijai"),
                "description": "Wan 2.2 T2V-A14B FP8 (MoE, Text-to-Video only, High Quality) [Kijai]",
                "size_gb": 18,
                "quantization": "FP8",
                "vram": "~18GB",
                "source": "Kijai"
            },

            # Kijai's Full Precision ComfyUI Models
            "TI2V-5B-Kijai": {
                "repo_id": "Kijai/WanVideo_comfy",
                "local_dir": str(self.models_dir / "Wan2.2-TI2V-5B-Kijai"),
                "description": "Wan 2.2 TI2V-5B (Unified T2V + I2V, 720P@24fps, Full Precision) [Kijai]",
                "size_gb": 30,
                "quantization": "FP16",
                "vram": "~24GB",
                "source": "Kijai"
            },

            # Official Wan-AI Models (Diffusers format)
            "TI2V-5B-Official": {
                "repo_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "local_dir": str(self.models_dir / "Wan2.2-TI2V-5B-Official"),
                "description": "Wan 2.2 TI2V-5B (Unified T2V + I2V, 720P@24fps, Official Release)",
                "size_gb": 30,
                "quantization": "FP16",
                "vram": "~24GB",
                "source": "Official"
            },
            "TI2V-A14B-Official": {
                "repo_id": "Wan-AI/Wan2.2-TI2V-A14B-Diffusers",
                "local_dir": str(self.models_dir / "Wan2.2-TI2V-A14B-Official"),
                "description": "Wan 2.2 TI2V-A14B (MoE, Unified T2V + I2V, Highest Quality, Official Release)",
                "size_gb": 60,
                "quantization": "FP16",
                "vram": "~32GB",
                "source": "Official"
            }
        }
    
    def _detect_models_directory(self) -> Path:
        """Detect the best models directory for the current installation"""
        # Try to find the webui models directory
        extension_root = Path(__file__).parent.parent.parent.parent
        
        # Option 1: webui/models/wan (standard installation)
        webui_models = extension_root.parent.parent / "models" / "wan"
        if webui_models.parent.exists():
            webui_models.mkdir(exist_ok=True)
            return webui_models
        
        # Option 2: Current working directory models/wan
        local_models = Path("models/wan")
        if local_models.parent.exists() or Path("models").exists():
            local_models.mkdir(parents=True, exist_ok=True)
            return local_models
        
        # Option 3: Extension directory models (fallback)
        extension_models = extension_root / "models" / "wan"
        extension_models.mkdir(parents=True, exist_ok=True)
        return extension_models
    
    def check_huggingface_cli(self) -> bool:
        """Check if huggingface-cli is available"""
        try:
            result = subprocess.run(
                ["huggingface-cli", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def install_huggingface_hub(self) -> bool:
        """Install huggingface_hub if not available"""
        try:
            print("📦 Installing huggingface_hub...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "huggingface_hub"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print("✅ huggingface_hub installed successfully")
                return True
            else:
                print(f"❌ Failed to install huggingface_hub: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Installation timed out")
            return False
        except Exception as e:
            print(f"❌ Installation error: {e}")
            return False
    
    def download_model(self, model_key: str, progress_callback=None) -> bool:
        """Download a specific model"""
        if model_key not in self.available_models:
            print(f"❌ Unknown model: {model_key}")
            return False
        
        model_info = self.available_models[model_key]
        local_dir = Path(model_info["local_dir"])
        
        # Check if model already exists
        if self.is_model_downloaded(model_key):
            print(f"✅ Model {model_key} already exists at {local_dir}")
            return True
        
        # Ensure huggingface-cli is available
        if not self.check_huggingface_cli():
            print("⚠️ huggingface-cli not found, installing huggingface_hub...")
            if not self.install_huggingface_hub():
                print("❌ Failed to install huggingface_hub")
                return False
        
        print(f"📥 Downloading {model_key} ({model_info['description']})...")
        print(f"   📂 From: {model_info['repo_id']}")
        print(f"   📁 To: {local_dir}")
        print(f"   💾 Size: ~{model_info['size_gb']}GB")
        
        # Create directory
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use huggingface-cli to download
            cmd = [
                "huggingface-cli", "download",
                model_info["repo_id"],
                "--local-dir", str(local_dir),
                "--local-dir-use-symlinks", "False"
            ]
            
            print(f"🚀 Running: {' '.join(cmd)}")
            
            # Run with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output
            for line in process.stdout:
                line = line.strip()
                if line:
                    print(f"   {line}")
                    if progress_callback:
                        progress_callback(line)
            
            process.wait()
            
            if process.returncode == 0:
                print(f"✅ Successfully downloaded {model_key}")
                return True
            else:
                print(f"❌ Download failed with return code {process.returncode}")
                return False
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            return False
    
    def is_model_downloaded(self, model_key: str) -> bool:
        """Check if a model is already downloaded"""
        if model_key not in self.available_models:
            return False
        
        model_info = self.available_models[model_key]
        local_dir = Path(model_info["local_dir"])
        
        # Check for essential files
        required_files = [
            "config.json"
        ]
        
        for file in required_files:
            if not (local_dir / file).exists():
                return False
        
        # Check for diffusion model files (single file OR multi-part)
        single_diffusion_file = local_dir / "diffusion_pytorch_model.safetensors"
        multi_part_index = local_dir / "diffusion_pytorch_model.safetensors.index.json"
        
        if single_diffusion_file.exists():
            # Single file model (1.3B)
            pass
        elif multi_part_index.exists():
            # Multi-part model (14B) - verify at least some files exist
            multi_part_files = list(local_dir.glob("diffusion_pytorch_model-*-of-*.safetensors"))
            if not multi_part_files:
                return False
        else:
            # No diffusion model found
            return False
        
        return True
    
    def get_model_path(self, model_key: str) -> Optional[str]:
        """Get the local path for a downloaded model"""
        if not self.is_model_downloaded(model_key):
            return None
        
        model_info = self.available_models[model_key]
        return str(Path(model_info["local_dir"]).absolute())
    
    def list_available_models(self) -> List[Dict]:
        """List all available models with download status"""
        models = []
        for key, info in self.available_models.items():
            models.append({
                "key": key,
                "description": info["description"],
                "size_gb": info["size_gb"],
                "downloaded": self.is_model_downloaded(key),
                "path": self.get_model_path(key) if self.is_model_downloaded(key) else None
            })
        return models
    
    def auto_download_recommended(self) -> Dict[str, str]:
        """Auto-download recommended model (TI2V-5B-FP8 for <16GB VRAM)"""
        results = {}

        # Download TI2V-5B-FP8 (Recommended for GPUs with <16GB VRAM)
        print("🎯 Auto-downloading Wan 2.2 TI2V-5B-FP8 (Unified T2V+I2V, 720p@24fps, <16GB VRAM)")
        if self.download_model("TI2V-5B-FP8"):
            model_path = self.get_model_path("TI2V-5B-FP8")
            results["t2v"] = model_path
            results["i2v"] = model_path  # TI2V handles both T2V and I2V
            print("✅ TI2V-5B-FP8 ready for unified T2V and I2V generation")
        else:
            print("❌ Failed to download TI2V-5B-FP8 model")
            print("💡 Trying full precision TI2V-5B as fallback...")
            if self.download_model("TI2V-5B"):
                model_path = self.get_model_path("TI2V-5B")
                results["t2v"] = model_path
                results["i2v"] = model_path
                print("✅ TI2V-5B (FP16) downloaded, but requires >24GB VRAM")

        return results
    
    def download_by_preference(self, prefer_size: str = "TI2V-5B", download_i2v: bool = False) -> Dict[str, str]:
        """Download Wan 2.2 TI2V model based on size preference"""
        results = {}

        # Determine which TI2V model to download (Wan 2.2 only)
        if "A14B" in prefer_size or "14B" in prefer_size:
            primary_model = "TI2V-A14B"
            print("✅ Wan 2.2 TI2V-A14B: MoE, unified T2V+I2V, 720p, highest quality")
        else:
            # Default to TI2V-5B (recommended)
            primary_model = "TI2V-5B"
            print("✅ Wan 2.2 TI2V-5B: Unified T2V+I2V, 720p@24fps, RTX 4090")

        # Download the TI2V model
        print(f"📥 Downloading {primary_model}...")
        if self.download_model(primary_model):
            model_path = self.get_model_path(primary_model)
            results["t2v"] = model_path
            results["i2v"] = model_path  # TI2V handles both T2V and I2V
            print(f"✅ {primary_model} ready for unified T2V and I2V generation")
        else:
            print(f"❌ Failed to download {primary_model}")

        return results

def download_wan_model(model_key: str) -> bool:
    """Convenience function to download a single model"""
    downloader = WanModelDownloader()
    return downloader.download_model(model_key)

def auto_setup_wan_models(prefer_size: str = "TI2V-5B") -> Dict[str, str]:
    """Auto-setup Wan models with size preference"""
    downloader = WanModelDownloader()
    return downloader.download_by_preference(prefer_size)

if __name__ == "__main__":
    # Test the downloader
    downloader = WanModelDownloader()
    
    print("🔍 Available models:")
    for model in downloader.list_available_models():
        status = "✅ Downloaded" if model["downloaded"] else "❌ Not downloaded"
        print(f"   {model['key']}: {model['description']} ({model['size_gb']}GB) - {status}")
    
    # Test download (uncomment to actually download)
    # print("\n📥 Testing download...")
    # downloader.download_model("1.3B T2V") 