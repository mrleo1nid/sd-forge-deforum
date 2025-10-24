# Flux ControlNet Integration Status

⚠️ **STATUS: FOUNDATION COMPLETE, NEEDS V2 REFACTOR (NOT READY FOR USE)**

## Current State (v1 - NOT VIABLE - Double VRAM usage)

### What Works ✅
- Flux ControlNet model loading (Canny and Depth)
- Preprocessing control images (Canny edge detection, Depth maps)
- Keyframe generation with ControlNet guidance
- Integration with Deforum's keyframe distribution system
- Multiple model providers (InstantX, XLabs-AI, BFL, Shakker Labs)

### Why v1 is NOT VIABLE ❌

**VRAM Usage:**
- Creates separate FluxControlNetPipeline from diffusers
- Loads full Flux model (~12GB) in addition to Forge's loaded model
- **Effectively doubles VRAM usage** - requires ~24GB minimum
- **Target users have 16GB VRAM** - this approach doesn't work
- **Must implement v2 (Forge-native)** to be usable

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

### The Challenge

Forge's Flux transformer (`backend/nn/flux.py:372-398`) does NOT have ControlNet parameters. Diffusers' Flux transformer accepts `controlnet_block_samples` and `controlnet_single_block_samples`, but Forge's implementation is missing this.

We need to inject control samples into Forge's Flux without duplicating the base model.

## Decision: Implement v2 (Forge-native integration)

**v1 is REJECTED** - double VRAM usage makes it unusable for target users (16GB VRAM).

**Must implement v2** to make this feature viable. This is a multi-session task.

## V2 Implementation Roadmap

### Phase 1: Research & Understanding ✅ COMPLETED

**1. Forge's ControlNet Backend** (`backend/patcher/controlnet.py`)
- `apply_controlnet_advanced()` works at lines 11-83
- Uses `ControlBase` class (lines 175-279) and `ControlNet` class (lines 282-358)
- Control model called at line 338: `self.control_model(x=x_noisy, hint=self.cond_hint, timesteps=timestep, context=context, y=y)`
- Control signals merged via `add_patched_controlnet()` into UNet
- ⚠️ **This system is SD/SDXL only - no Flux support**

**2. FluxControlNetModel Internals** (`diffusers/models/controlnets/controlnet_flux.py`)
- Forward signature at lines 213-280:
  ```python
  forward(hidden_states, controlnet_cond, conditioning_scale=1.0,
          encoder_hidden_states, pooled_projections, timestep,
          img_ids, txt_ids, guidance, ...)
  ```
- Returns: `FluxControlNetOutput(controlnet_block_samples, controlnet_single_block_samples)`
- ✅ **Can be called directly without full pipeline** - just need proper inputs

**3. How Diffusers Uses Control Samples** (`diffusers/pipelines/flux/pipeline_flux_controlnet.py`)
- Line 1082: Computes control samples via `self.controlnet(...)`
- Lines 1108-1127: Passes `controlnet_block_samples` and `controlnet_single_block_samples` to `self.transformer()`
- Control samples are injected into transformer's double_blocks and single_blocks

**4. Forge's Flux Architecture** (`backend/nn/flux.py`)
- Main forward at lines 400-422
- Inner forward at lines 372-398 with signature: `inner_forward(img, img_ids, txt, txt_ids, timesteps, y, guidance=None)`
- Line 388: `for block in self.double_blocks:` - ⚠️ **NO ControlNet injection here**
- Line 391: `for block in self.single_blocks:` - ⚠️ **NO ControlNet injection here**
- ❌ **Critical finding**: Forge's Flux transformer is missing `controlnet_block_samples` and `controlnet_single_block_samples` parameters

### Phase 2: Minimal Integration ⏳ IN PROGRESS

**1. Load Only ControlNet Model** ✅ DONE
- Created `flux_controlnet_v2.py` with `FluxControlNetV2Manager` class
- Loads **only** FluxControlNetModel (~3.6GB), not full pipeline
- No duplicate Flux model loaded (saves ~12GB)
- Model caching via existing `flux_controlnet_models.py`

**2. Compute Control Samples** ✅ DONE
- Preprocess control image (Canny/Depth) - reuses v1 preprocessors ✓
- `compute_control_samples()` method calls `FluxControlNetModel.forward()` directly
- Returns `(controlnet_block_samples, controlnet_single_block_samples)` tuples
- Ready to inject into Forge's Flux transformer

**3. Patch Forge's Flux Transformer** ✅ DONE
- Added `patch_forge_flux_controlnet()` to `diffusers_compat_patch.py`
- Runtime patches `Flux.inner_forward()` to accept ControlNet parameters
- Runtime patches `Flux.forward()` to pass parameters through
- Injects control after each double_block and single_block (matches diffusers)
- Applied automatically at extension init via `apply_all_patches()`
- ✅ No Forge source file modifications needed!

**4. Wire Control into Generation Pipeline** ⏳ TODO
- Need to connect V2 manager to Deforum's generation flow
- Compute control samples in `generate_with_flux_controlnet()`
- Pass `controlnet_block_samples` and `controlnet_single_block_samples` to Forge
- Options for passing control:
  - Via `model_options` with custom conditioning modifier
  - Via `**extra_conds` in k_model.py apply_model()
  - Via Forge's processing pipeline hooks

### Phase 3: Testing & Refinement

**1. Test with 16GB VRAM**
- Verify no VRAM duplication
- Confirm Forge's Flux + ControlNet model fits
- Measure performance

**2. Refine Control Application**
- Tune strength/conditioning parameters
- Test Canny and Depth modes
- Verify visual quality

**3. Clean Up**
- Remove v1 pipeline code
- Update documentation
- Add usage examples

### Phase 1 Questions - ANSWERED ✅

1. **Can we call FluxControlNetModel.forward() directly?**
   - ✅ YES - Found at `diffusers/models/controlnets/controlnet_flux.py:213-280`
   - Inputs: `hidden_states, controlnet_cond, conditioning_scale, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance`
   - Outputs: `FluxControlNetOutput(controlnet_block_samples, controlnet_single_block_samples)`

2. **Where in Forge's Flux pipeline can we inject control?**
   - ✅ Need to inject during transformer forward pass
   - Specifically: `backend/nn/flux.py:388` (double_blocks) and line 391 (single_blocks)
   - Diffusers injects control samples directly into transformer blocks

3. **Does Forge's backend support Flux at all?**
   - ✅ YES - Flux implementation at `backend/nn/flux.py`
   - ❌ NO ControlNet support - transformer missing controlnet parameters
   - Need to add ControlNet capability ourselves

4. **Can we reuse any code from archive/crazy-flux-controlnet-monkeypatch?**
   - ⚠️ That branch was 3958 lines of complex monkey-patching
   - User preference: "we like to do it better and avoid patching Forge from its own extension if possible"
   - Better to implement cleanly with new knowledge

### Success Criteria

- ✅ Loads Forge's Flux model (already loaded)
- ✅ Loads ControlNet model only (~3.6GB)
- ✅ Total VRAM < 16GB
- ✅ Generates with ControlNet guidance
- ✅ Quality comparable to diffusers pipeline
- ✅ No Forge backend monkey-patching (if possible)

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
