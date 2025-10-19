# Wan UI Cleanup TODO

## Current Status
✅ Backend fully integrated:
- Wan FLF2V works in experimental render core
- Three modes: Flux/tween, Flux/Wan, Flux/tween/Wan
- VRAM optimization settings functional
- FLF2V chaining for long sections

## Remaining UI Work

### 1. Simplify Wan Tab (HIGH PRIORITY)
**Current:** 800+ lines, standalone workflow UI
**Target:** ~100 lines, just model/VRAM settings

**Remove/Hide:**
- ❌ Separate "Wan Video Prompts" section (lines 1128-1223)
- ❌ "Generate Wan Video" button (lines 1244-1260)
- ❌ "Deforum Integration" explanation accordion (lines 1225-1241)
- ❌ Movement analysis buttons (lines 1162-1201)
- ❌ "Essential Settings" accordion (lines 1263-1302)

**Keep:**
- ✅ Intro markdown explaining integration
- ✅ Model download buttons
- ✅ Model selection dropdown
- ✅ VRAM Optimization accordion

### 2. Move Qwen to Prompts Tab
**From:** Wan tab lines 1304-1316
**To:** Prompts tab

Add accordion in Prompts tab:
```
with gr.Accordion("🧠 AI Prompt Enhancement (Qwen)", open=False):
    - Qwen model selector
    - Language selector  
    - Auto-download toggle
    - Enhance button (works on current prompts)
```

### 3. Update Default Settings
Check if default_settings.txt needs:
- enable_wan_flf2v: False
- wan_flf2v_chunk_size: 81

### 4. Deprecate Hybrid Video Tab
Either:
- Set `visible=False` on the TabItem
- Or replace with deprecation notice

### 5. Deprecate Old Depth Models
In depth model UI, hide/deprecate:
- MiDaS
- AdaBins  
- LeReS
- ZoeDepth

Keep only:
- Depth-Anything-V2 (default)

## Implementation Plan

### Phase 1: Wan Tab Cleanup
1. Read full get_tab_wan function (lines 1124-1919)
2. Wrap deprecated sections in `visible=False` accordion
3. Rewrite to match simplified structure
4. Test that model selection still works

### Phase 2: Move Qwen
1. Find Prompts tab location
2. Add Qwen accordion after prompts textbox
3. Wire up enhance button to modify prompts
4. Remove from Wan tab

### Phase 3: Deprecations
1. Hide hybrid tab
2. Hide old depth models
3. Update default settings

### Phase 4: Testing
1. Test main Generate button with Wan FLF2V enabled
2. Test model auto-discovery
3. Test VRAM settings apply correctly
4. Test Qwen enhancement from Prompts tab

## Notes
- All backend code is functional
- UI changes are non-breaking (just hiding/moving)
- Can be done incrementally
- May want user feedback before removing vs hiding

