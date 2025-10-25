# Remaining Tasks After Render Mode Refactor

## Completed ✅
1. **Flux Availability Check** - Already implemented in `deforum/utils/system/flux_check.py`
   - Shows blocker message if Flux not selected
   - Prevents extension from loading without Flux

## In Progress 🔧

### 2. Remove vid2depth.py
**Status**: Ready to remove
**Reason**: Uses legacy MiDaS/AdaBins models that were removed. Marked as "legacy parameter, no longer used with Depth-Anything V2"
**Files to remove/modify**:
- `deforum/depth/vid2depth.py` - Remove entire file
- `deforum/depth/__init__.py` - Remove vid2depth imports
- `deforum/ui/ui_elements.py` - Remove Vid2Depth tab (lines ~2397-2431)
- `deforum/ui/gradio_funcs.py` - Remove upload_vid_to_depth function

## Pending 📋

### 3. Flux ControlNet Fallback
**Status**: Needs investigation
**Issue**: User reported logs showing "fallback that loads two flux models"
**Current implementation**: Only loads FluxControlNetModel (~3.6GB), no fallback found in code
**Action**: May already be fixed - need to check runtime logs

### 4. Masking Integration
**Status**: Needs design
**Current**: Masking code exists but not well integrated in UI
**Goal**:
- Create dedicated "Masking" tab
- Text masking would be a great feature
- Possibly at same level as 3D Depth, Shakify tabs

### 5. Link to Forge Persist Settings
**Status**: Easy win
**Current**: Save settings section has buttons to save settings.txt
**Goal**: Add note/link directing users to Forge's built-in persist settings
**Location**: Near save settings buttons in UI

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

## Priority Order
1. Remove vid2depth (cleanup, prevents confusion)
2. Check ControlNet fallback in runtime logs
3. Link to Forge persist settings (5 min fix)
4. Masking UI integration (moderate effort)
5. Deeper Forge integration (research phase)
6. Audio extraction (future feature, document for later)
