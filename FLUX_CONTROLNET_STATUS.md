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

## Decision: Implement v2 (Forge-native integration)

**v1 is REJECTED** - double VRAM usage makes it unusable for target users (16GB VRAM).

**Must implement v2** (Option C - Hybrid approach) to make this feature viable. This is a multi-session task.

## V2 Implementation Roadmap

### Phase 1: Research & Understanding (Next Session)

**1. Study Forge's ControlNet Backend** (`backend/patcher/controlnet.py`)
- How does `apply_controlnet_advanced()` work for SD/SDXL?
- How are control hints injected into UNet?
- What is the `set_cond_hint()` API?
- How does `add_patched_controlnet()` work?

**2. Study FluxControlNetModel Internals**
- How does FluxControlNetModel compute control hints?
- What's the forward pass signature?
- Can we extract control tensors without full pipeline?
- What conditioning format does Flux expect?

**3. Understand Forge's Flux Loading**
- How does Forge load `flux1-dev-bnb-nf4-v2.safetensors`?
- Where is the Flux transformer in `p.sd_model.forge_objects`?
- Can we access the Flux UNet/transformer directly?

### Phase 2: Minimal Integration (After Research)

**1. Load Only ControlNet Model**
- Modify `flux_controlnet_models.py` to load **only** FluxControlNetModel (~3.6GB)
- Don't create FluxControlNetPipeline (saves ~12GB)
- Cache the ControlNet model between frames

**2. Compute Control Hints**
- Preprocess control image (Canny/Depth) - **already done** ✓
- Pass through FluxControlNetModel forward pass
- Extract control tensors/embeddings
- Format for Forge injection

**3. Inject into Forge's Processing**
- Option A: Modify `generate.py` to add control to Forge's Flux processing
- Option B: Create Flux-specific ControlNet patcher in Forge backend
- Option C: Inject control tensors directly into `p.sd_model.forge_objects.unet`

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

### Open Questions for Next Session

1. **Can we call FluxControlNetModel.forward() directly?**
   - Need to check diffusers source code
   - What inputs does it expect?
   - What outputs does it produce?

2. **Where in Forge's Flux pipeline can we inject control?**
   - During text encoding?
   - During transformer forward pass?
   - Via custom attention injection?

3. **Does Forge's backend support Flux at all?**
   - Check `backend/nn/` for Flux-specific code
   - May need to implement from scratch

4. **Can we reuse any code from archive/crazy-flux-controlnet-monkeypatch?**
   - Was it attempting v2-style integration?
   - Any useful patterns to extract?

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
