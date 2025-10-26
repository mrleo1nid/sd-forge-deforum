#!/bin/bash

# download-all-models.sh
# Downloads all required models for Deforum extension at once
# This ensures a complete installation and tests all download hooks

set -e  # Exit on error

echo "========================================"
echo "Deforum Model Download Script"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory (extension root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Navigate to Forge root (two levels up)
FORGE_ROOT="$(cd ../.. && pwd)"
cd "$FORGE_ROOT"

echo -e "${BLUE}Working from Forge root: $FORGE_ROOT${NC}"
echo ""

# Create model directories
echo -e "${YELLOW}Creating model directories...${NC}"
mkdir -p models/Deforum/film_interpolation
mkdir -p models/wan
mkdir -p models/qwen
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Check for HuggingFace CLI
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${RED}❌ huggingface-cli not found!${NC}"
    echo ""
    echo "Please install it with:"
    echo "  pip install huggingface-hub"
    echo ""
    exit 1
fi

# Function to download with progress
download_file() {
    local url=$1
    local dest=$2
    local name=$3

    echo -e "${YELLOW}Downloading $name...${NC}"
    python -c "
import sys
from torch.hub import download_url_to_file
download_url_to_file('$url', '$dest', progress=True)
"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $name downloaded successfully${NC}"
    else
        echo -e "${RED}❌ Failed to download $name${NC}"
        return 1
    fi
}

# =====================================
# 1. FILM Interpolation Model
# =====================================
echo -e "${BLUE}=== FILM Interpolation Model ===${NC}"
FILM_PATH="models/Deforum/film_interpolation/film_net_fp16.pt"
if [ -f "$FILM_PATH" ]; then
    echo -e "${GREEN}✓ FILM model already exists${NC}"
else
    download_file \
        "https://github.com/hithereai/frame-interpolation-pytorch/releases/download/film_net_fp16.pt/film_net_fp16.pt" \
        "$FILM_PATH" \
        "FILM model (film_net_fp16.pt)"
fi
echo ""

# =====================================
# 2. Wan Models (HuggingFace)
# =====================================
echo -e "${BLUE}=== Wan AI Video Models ===${NC}"
echo "Choose which Wan models to download:"
echo "  1) FLF2V-14B (Required for FLF2V interpolation, ~14GB)"
echo "  2) TI2V-5B (Recommended for T2V/I2V, 24GB VRAM, ~5GB)"
echo "  3) TI2V-A14B (Highest quality MoE, 32GB+ VRAM, ~14GB)"
echo "  4) All models (Downloads all 3)"
echo "  5) Skip Wan models"
read -p "Enter choice [1-5]: " wan_choice

case $wan_choice in
    1|4)
        echo -e "${YELLOW}Downloading Wan2.1-FLF2V-14B...${NC}"
        huggingface-cli download Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers \
            --local-dir models/wan/Wan2.1-FLF2V-14B \
            --resume-download
        echo -e "${GREEN}✓ FLF2V-14B downloaded${NC}"
        ;&  # Fall through if choice was 4
esac

case $wan_choice in
    2|4)
        echo -e "${YELLOW}Downloading Wan2.2-TI2V-5B...${NC}"
        huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
            --local-dir models/wan/Wan2.2-TI2V-5B \
            --resume-download
        echo -e "${GREEN}✓ TI2V-5B downloaded${NC}"
        ;&  # Fall through if choice was 4
esac

case $wan_choice in
    3|4)
        echo -e "${YELLOW}Downloading Wan2.2-TI2V-A14B...${NC}"
        huggingface-cli download Wan-AI/Wan2.2-TI2V-A14B-Diffusers \
            --local-dir models/wan/Wan2.2-TI2V-A14B \
            --resume-download
        echo -e "${GREEN}✓ TI2V-A14B downloaded${NC}"
        ;;
    5)
        echo -e "${YELLOW}Skipping Wan models${NC}"
        ;;
esac
echo ""

# =====================================
# 3. Qwen AI Prompt Enhancement Models
# =====================================
echo -e "${BLUE}=== Qwen Prompt Enhancement Models ===${NC}"
echo "Choose which Qwen model to download (for AI prompt enhancement):"
echo "  1) Qwen2.5-3B-Instruct (Recommended for low VRAM, ~3GB)"
echo "  2) Qwen2.5-7B-Instruct (Better quality, ~7GB)"
echo "  3) Qwen2.5-14B-Instruct (Best quality, 32GB+ VRAM, ~14GB)"
echo "  4) All models (Downloads all 3)"
echo "  5) Skip Qwen models"
read -p "Enter choice [1-5]: " qwen_choice

case $qwen_choice in
    1|4)
        echo -e "${YELLOW}Downloading Qwen2.5-3B-Instruct...${NC}"
        huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
            --local-dir models/qwen/Qwen2.5-3B-Instruct \
            --resume-download
        echo -e "${GREEN}✓ Qwen2.5-3B-Instruct downloaded${NC}"
        ;&  # Fall through if choice was 4
esac

case $qwen_choice in
    2|4)
        echo -e "${YELLOW}Downloading Qwen2.5-7B-Instruct...${NC}"
        huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
            --local-dir models/qwen/Qwen2.5-7B-Instruct \
            --resume-download
        echo -e "${GREEN}✓ Qwen2.5-7B-Instruct downloaded${NC}"
        ;&  # Fall through if choice was 4
esac

case $qwen_choice in
    3|4)
        echo -e "${YELLOW}Downloading Qwen2.5-14B-Instruct...${NC}"
        huggingface-cli download Qwen/Qwen2.5-14B-Instruct \
            --local-dir models/qwen/Qwen2.5-14B-Instruct \
            --resume-download
        echo -e "${GREEN}✓ Qwen2.5-14B-Instruct downloaded${NC}"
        ;;
    5)
        echo -e "${YELLOW}Skipping Qwen models${NC}"
        ;;
esac
echo ""

# =====================================
# Summary
# =====================================
echo ""
echo -e "${GREEN}========================================"
echo "✅ Model Download Complete!"
echo "========================================${NC}"
echo ""
echo "Downloaded models are located in:"
echo "  • FILM: models/Deforum/film_interpolation/"
echo "  • Wan: models/wan/"
echo "  • Qwen: models/qwen/"
echo ""
echo -e "${BLUE}Note:${NC} Depth models (Depth-Anything V2) will be auto-downloaded"
echo "on first use. Gifski and Real-ESRGAN binaries are also auto-downloaded."
echo ""
echo -e "${YELLOW}You can now use Deforum with Flux + Interpolation mode!${NC}"
echo ""
