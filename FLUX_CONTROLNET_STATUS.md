# Flux ControlNet Integration Status

## Current State (v1 - Working but with limitations)

### What Works ✅
- Flux ControlNet model loading (Canny and Depth)
- Preprocessing control images (Canny edge detection, Depth maps)
- Keyframe generation with ControlNet guidance
- Integration with Deforum's keyframe distribution system
- Multiple model providers (InstantX, XLabs-AI, BFL, Shakker Labs)

### Limitations ⚠️

**VRAM Usage:**
- Creates separate FluxControlNetPipeline from diffusers
- Loads full Flux model (~12GB) in addition to Forge's loaded model
- Effectively doubles VRAM usage for Flux
- Not ideal, but functional

**Authentication Required:**
1. Run: `huggingface-cli login`
2. Enter your HuggingFace token
3. Accept FLUX.1-dev license at: https://huggingface.co/black-forest-labs/FLUX.1-dev

Without this, you'll get: `401 Client Error: Unauthorized`

### How to Use

1. **Enable in UI**: Navigate to **3D Depth tab → 🌐 Flux ControlNet accordion**
2. **Configure**:
   - Enable Flux ControlNet: ☑️
   - ControlNet Type: `canny` or `depth`
   - Model: `instantx` (recommended), `xlabs`, `bfl`, `shakker`
   - Strength: 0.7 (default)
3. **First use**: Models will download from HuggingFace (~3.6GB for ControlNet)
4. **Generate**: Only applies to **keyframes** (not tween frames)

### Architecture

**Current (v1):**
```
User → Deforum → diffusers FluxControlNetPipeline → Generate
                              ↓
                    Downloads/Loads Flux model from HF
```

**Future (v2 - Forge-native):**
```
User → Deforum → Forge's loaded Flux + ControlNet model → Generate
                              ↓
                    Reuses Forge's model (no duplication)
```

## Future Direction (v2 - Forge-native integration)

### Goals
- Load ONLY FluxControlNetModel (~3.6GB)
- Reuse Forge's already-loaded Flux transformer
- No VRAM duplication
- No HuggingFace authentication needed
- Integrate with Forge's backend patching system

### Challenges
1. **Format incompatibility**:
   - Forge: Single `.safetensors` file (`flux1-dev-bnb-nf4-v2.safetensors`)
   - Diffusers: Full model repo with `config.json`, multiple files
   - No direct conversion path

2. **Forge backend limitations**:
   - No Flux ControlNet support in `backend/patcher/controlnet.py`
   - Would need to implement Flux-specific ControlNet patcher
   - Forge project is semi-abandoned (no official updates expected)

3. **Complexity**:
   - Need to understand Flux transformer internals
   - Need to understand how ControlNet conditioning works for Flux
   - Need to patch Forge's backend from extension (not ideal)

### Possible Approaches

**Option A: Wait for community support**
- Monitor for community Flux ControlNet implementations for Forge
- Unlikely given Forge's abandoned status

**Option B: Implement Forge backend patcher**
- Study ComfyUI's Flux ControlNet implementation
- Port to Forge's backend architecture
- Significant effort, requires deep understanding

**Option C: Hybrid approach**
- Load ControlNet model only (not full pipeline)
- Manually compute control hints
- Inject into Forge's Flux generation somehow
- Requires experimentation

**Option D: Accept current state**
- v1 works, just uses more VRAM
- Document authentication requirement
- Focus on other features
- Revisit when we have more time/knowledge

## Decision

For now, **proceeding with Option D** (accept current state):
- Feature is functional
- VRAM usage is acceptable for most users (24GB+ GPUs)
- Authentication is one-time setup
- Can optimize later when we better understand architecture

Users with limited VRAM can disable Flux ControlNet and use standard Flux generation.

## Related Files

- `scripts/deforum_helpers/flux_controlnet.py` - Main FluxControlNetManager
- `scripts/deforum_helpers/flux_controlnet_models.py` - Model loading
- `scripts/deforum_helpers/flux_controlnet_preprocessors.py` - Image preprocessing
- `scripts/deforum_helpers/rendering/util/flux_controlnet_integration.py` - Deforum integration
- `scripts/deforum_helpers/rendering/util/call/gen.py` - Generation routing
- `scripts/deforum_helpers/args.py:638-699` - Settings/arguments
- `scripts/deforum_helpers/ui_elements.py:619-639` - UI controls

## References

- Previous attempt: `archive/crazy-flux-controlnet-monkeypatch` branch
- Black Forest Labs Flux ControlNet: https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev
- InstantX ControlNet: https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny
- Diffusers FluxControlNet: https://huggingface.co/docs/diffusers/api/pipelines/flux
