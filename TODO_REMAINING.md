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

### 4. Masking Integration
**Completed**: Implemented in commits 87bc0e9 and 11f68be
**Location**: New main-level "Masking" tab (tab #7, after Shakify, before Wan Models)

**Implementation**:
Created new `get_tab_masking()` function with 4 sub-tabs:
1. **📝 Basic Masks** - File upload, alpha channel, invert, overlay, blur/contrast/brightness adjustments
2. **✨ Text Masking (CLIPSeg)** - AI text-to-mask with comprehensive guide and examples
   - Syntax: `<cat>`, `<sky>`, `<person's face>`, `<armor>`
   - Uses ViT-B/16 CLIPSeg (auto-downloads on first use)
   - Integration with composable masks
3. **🔧 Composable Masks** - Boolean mask operations with full documentation
   - Operators: `&` (AND), `|` (OR), `^` (XOR), `!` (NOT), `\` (DIFFERENCE)
   - Mask types: `{variables}`, `[files]`, `<text>`
   - Example: `(({human_mask} & [mask1.png]) ^ <apple>)`
4. **🎬 Video & Human** - Animated video masks + RobustVideoMatting AI human detection
   - Video masks: Per-frame animated sequences for rotoscoping
   - Human masks: Use `{human_mask}` in expressions for automatic detection

**Removed**:
- Hidden "Composable Mask scheduling" accordion from Run tab
- Hidden "Mask Init" tab from Init tab
- Duplicate video mask components from Video Init tab

**Files modified**:
- `deforum/ui/ui_elements.py`: New `get_tab_masking()` + helper, removed hidden sections
- `deforum/ui/ui_left.py`: Import and integrate masking tab

**Result**: Powerful masking features are now discoverable with comprehensive documentation

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
1. ✅ ~~Remove vid2depth~~ - COMPLETED (commits ad839a5, 1a76fd4)
2. ✅ ~~Link to Forge persist settings~~ - COMPLETED (commit 7226427)
3. ✅ ~~Implement Masking UI tab~~ - COMPLETED (commits 87bc0e9, 11f68be)
   - Exposed CLIPSeg text-to-mask feature (user requested)
   - Unified scattered masking UI with comprehensive docs
   - Made powerful hidden features discoverable
4. **Check ControlNet fallback in runtime logs** - Next priority (may already be fixed)
5. Deeper Forge integration (research phase)
6. Audio extraction (future feature, document for later)
