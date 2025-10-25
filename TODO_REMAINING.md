# Remaining Tasks After Render Mode Refactor

## Completed ✅

### 1. Flux Availability Check
Already implemented in `deforum/utils/system/flux_check.py`
- Shows blocker message if Flux not selected
- Prevents extension from loading without Flux

### 2. Remove vid2depth.py
**Completed**: Removed in commits ad839a5 and 1a76fd4
**Reason**: Uses legacy MiDaS/AdaBins models that were removed
**Files removed/modified**:
- ✅ `deforum/depth/vid2depth.py` - Deleted entire file
- ✅ `deforum/depth/__init__.py` - Removed vid2depth imports
- ✅ `deforum/ui/ui_elements.py` - Removed Vid2Depth tab, fixed imports
- ✅ `deforum/ui/gradio_funcs.py` - Removed upload_vid_to_depth function

### 3. Link to Forge Persist Settings
**Completed**: Added in commit 7226427
**Location**: Run tab → "Batch Mode, Resume and more" accordion
**Implementation**: Blue info box directing users to Settings → Defaults in Forge
**Result**: Users now understand difference between Forge persist settings and Deforum batch mode

## In Progress 🔧

### 4. Masking Integration
**Status**: Investigation complete - ready for implementation
**Findings**: Extensive masking features exist but are HIDDEN or poorly integrated

**Existing Features Found**:
1. **Composable Mask Scheduling** - `deforum/ui/ui_elements.py:441` - accordion with `visible=False`
   - Boolean operations: AND (&), OR (|), XOR (^), NOT (!), DIFFERENCE (\)
   - Three mask types:
     - Variable masks `{name}` (e.g., `{human_mask}`, `{video_mask}`)
     - File masks `[path.png]` - load from files
     - **Word masks `<text>`** - CLIPSeg text-based masking (e.g., `<cat>`, `<armor>`)
   - Example: `(({human_mask} & [mask1.png]) ^ <apple>)`
   - Implementation: `deforum/core/masking/composable.py`

2. **CLIPSeg Text-to-Mask** - `deforum/core/masking/word.py`
   - Generate masks from text descriptions
   - Uses ViT-B/16 CLIPSeg model
   - Fully functional but NO UI
   - Syntax: `<cat>`, `<person's face>`, `<sky>`

3. **Human Detection Masking** - `deforum/core/masking/human.py`
   - Auto-detect humans using RobustVideoMatting
   - Used as `{human_mask}` in composable expressions
   - No UI for configuration

4. **Basic Mask Init** - `deforum/ui/ui_elements.py:704` - tab with `visible=False`
   - File upload, alpha channel, invert, overlay
   - Blur, contrast, brightness adjustments
   - Full resolution mask with padding

5. **Video Masks** - Visible in Video Init tab
   - Per-frame video mask sequences

**Proposed Solution**:
Create new **"Masking"** main tab (same level as "3D Depth", "Shakify") with 4 sub-tabs:
- **Basic Masks**: Unhide existing mask init UI (file upload, alpha, invert, overlay, adjustments)
- **Text Masking (CLIPSeg)**: New UI for text-to-mask with examples and preview
- **Composable Masks**: Unhide existing composable mask scheduler with guide/examples
- **Video & Human Masks**: Video mask paths, human detection toggle, variable masks

**Benefits**:
- Makes powerful existing features discoverable
- Text masking is unique feature (user specifically requested)
- Organizes scattered masking UI into coherent section
- No new backend code needed - just UI reorganization

## Pending 📋

### 5. Flux ControlNet Fallback
**Status**: Needs investigation
**Issue**: User reported logs showing "fallback that loads two flux models"
**Current implementation**: Only loads FluxControlNetModel (~3.6GB), no fallback found in code
**Action**: May already be fixed - need to check runtime logs

### 6. Deeper Forge Integration
**Status**: Ideas phase
**Potential improvements**:
- Better model management integration
- Shared VRAM optimization
- Unified checkpoint selector
- Better memory management coordination

### 7. Audio Event Extraction (Future)
**Status**: Future feature
**Current**: Requires parseq setup
**Goal**: Extract audio events and synchronize prompts automatically
**Use case**: Music-driven animations without manual parseq configuration

## Priority Order (Updated)
1. ✅ ~~Remove vid2depth~~ - COMPLETED
2. ✅ ~~Link to Forge persist settings~~ - COMPLETED
3. **Implement Masking UI tab** - Ready to implement (investigation complete)
   - High value: Makes powerful hidden features discoverable
   - Text masking (CLIPSeg) is unique feature user requested
   - Moderate effort: Mostly UI reorganization, backend already exists
4. Check ControlNet fallback in runtime logs (may already be fixed)
5. Deeper Forge integration (research phase)
6. Audio extraction (future feature, document for later)
