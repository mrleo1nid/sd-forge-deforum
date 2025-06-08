from deforum.config.args import DeforumOutputArgs
from deforum.config.arg_transformations import get_component_names, get_settings_component_names
from modules.shared import opts, state
from modules.ui import create_output_panel, wrap_gradio_call
from modules.call_queue import wrap_gradio_gpu_call
from deforum.core.run_deforum import run_deforum
from deforum.config.settings import save_settings, load_all_settings, load_video_settings, get_default_settings_path, update_settings_path
from deforum.utils.core_utilities import get_deforum_version, get_commit_date, debug_print
from .main_interface_panels import setup_deforum_left_side_ui
from deforum_extend_paths import deforum_sys_extend
import gradio as gr

def on_ui_tabs():
    # extend paths using sys.path.extend so we can access all of our files and folders
    deforum_sys_extend()
    
    try:
        # set text above generate button
        style = '"text-align:center;font-weight:bold;margin-bottom:0em"'
        extension_url = "https://github.com/Tok/sd-forge-deforum"
        link = f"<a href='{extension_url}' target='_blank'>Zirteqs Fluxabled Fork</a>"
        extension_name = f"{link} of the Deforum Fork for WebUI Forge"

        commit_info = f"Git commit: {get_deforum_version()}"
        i1_store_backup = f"<p style={style}>{extension_name} - Version: {get_commit_date()} | {commit_info}</p>"
        i1_store = i1_store_backup

        debug_print("Creating Gradio interface...")
        
        try:
            debug_print("Creating Deforum interface with proper structure...")
            
            with gr.Blocks(analytics_enabled=False) as deforum_interface:
                debug_print("Inside Gradio Blocks context...")
                
                # Header
                style = '"text-align:center;font-weight:bold;margin-bottom:0em"'
                extension_url = "https://github.com/Tok/sd-forge-deforum"
                link = f"<a href='{extension_url}' target='_blank'>Zirteqs Fluxabled Fork</a>"
                extension_name = f"{link} of the Deforum Fork for WebUI Forge"
                commit_info = f"Git commit: {get_deforum_version()}"
                i1_store_backup = f"<p style={style}>{extension_name} - Version: {get_commit_date()} | {commit_info}</p>"
                
                with gr.Row(elem_id='deforum_progress_row', equal_height=False, variant='compact'):
                    with gr.Column(scale=1, variant='panel'):
                        # Try to set up the left side UI, but catch any errors
                        try:
                            debug_print("Setting up left side UI...")
                            components = setup_deforum_left_side_ui()
                            debug_print(f"Left side UI setup completed, got {len(components) if isinstance(components, dict) else 0} components")
                        except Exception as e:
                            print(f"⚠️ Warning: Could not set up full left side UI: {e}")
                            components = {}
                            # Create minimal fallback components
                            with gr.HTML("<h3>⚠️ Basic Deforum Interface</h3>"):
                                pass
                            with gr.HTML("<p>Some components could not load. Using simplified interface.</p>"):
                                pass
                        
                    with gr.Column(scale=1, variant='compact'):
                        # Right side - video display and controls
                        with gr.Row(variant='compact'):
                            btn = gr.Button("Click here after generation to show the video")
                            close_btn = gr.Button("Close the video", visible=False)
                        
                        with gr.Row(variant='compact'):
                            i1 = gr.HTML(i1_store_backup, elem_id='deforum_header')
                        
                        # Control buttons
                        id_part = 'deforum'
                        with gr.Row(elem_id=f"{id_part}_generate_box", variant='compact'):
                            skip = gr.Button('Pause/Resume', elem_id=f"{id_part}_skip", visible=False)
                            interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt", visible=True)
                            submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')
                        
                        # Output panel
                        try:
                            res = create_output_panel("deforum", opts.outdir_img2img_samples)
                            deforum_gallery = res.gallery
                            generation_info = res.generation_info
                            html_info = res.html_log
                        except Exception as e:
                            print(f"⚠️ Warning: Could not create output panel: {e}")
                            deforum_gallery = gr.Gallery(label="Generated Images")
                            generation_info = gr.Textbox(label="Generation Info")
                            html_info = gr.HTML()
                        
                        # Settings controls
                        with gr.Row(variant='compact'):
                            settings_path = gr.Textbox(get_default_settings_path(), elem_id='deforum_settings_path', label="Settings File")
                        with gr.Row(variant='compact'):
                            save_settings_btn = gr.Button('Save Settings')
                            load_settings_btn = gr.Button('Load All Settings')
                            load_video_settings_btn = gr.Button('Load Video Settings')
                
                # Set up basic event handlers safely
                debug_print("Setting up essential event handlers...")
                
                # Simple interrupt handler - only set up if component exists and has _id
                try:
                    if hasattr(interrupt, '_id') and hasattr(state, 'interrupt'):
                        interrupt.click(fn=lambda: state.interrupt(), inputs=None, outputs=None)
                        debug_print("Interrupt handler connected")
                except Exception as e:
                    print(f"⚠️ Could not connect interrupt handler: {e}")
                
                # Simple skip handler - only set up if component exists and has _id
                try:
                    if hasattr(skip, '_id') and hasattr(state, 'skip'):
                        skip.click(fn=lambda: state.skip(), inputs=None, outputs=None)
                        debug_print("Skip handler connected")
                except Exception as e:
                    print(f"⚠️ Could not connect skip handler: {e}")
                
                # FIXED: Connect the Generate button to run_deforum using named arguments
                try:
                    debug_print("Setting up Generate button with named arguments...")
                    
                    def create_named_run_wrapper():
                        """
                        Create a wrapper that converts component values to named arguments.
                        Uses DYNAMIC COMPONENT DISCOVERY to avoid missing component issues.
                        """
                        def named_run_deforum(*args):
                            # Convert positional args to named kwargs
                            kwargs = {'job_id': 'deforum_job'}
                            
                            # VERIFICATION: Confirm new dynamic system is running
                            print("🚀 DYNAMIC COMPONENT DISCOVERY SYSTEM ACTIVE 🚀")
                            
                            # WORKAROUND: Load settings directly to get correct values
                            # This bypasses the Gradio component caching issue
                            try:
                                from ..config.settings import get_webui_settings_path
                                import json
                                import os
                                settings_path = get_webui_settings_path()
                                if os.path.exists(settings_path):
                                    with open(settings_path, 'r') as f:
                                        settings_data = json.load(f)
                                    print(f"🔧 Loaded settings from {settings_path} as fallback")
                                    print(f"   Settings W={settings_data.get('W')}, H={settings_data.get('H')}, strength={settings_data.get('strength')}")
                                else:
                                    settings_data = {}
                                    print(f"⚠️ Settings file not found: {settings_path}")
                            except Exception as e:
                                settings_data = {}
                                print(f"⚠️ Error loading settings: {e}")
                            
                            # DYNAMIC DISCOVERY: Only use components that actually exist in the UI
                            # This eliminates the misalignment caused by missing components
                            actual_component_order = []
                            component_names = get_component_names()
                            
                            # Build list of components that actually exist and have valid gradio _id
                            for name in component_names:
                                if name in components and components[name] is not None:
                                    comp = components[name]
                                    if hasattr(comp, '_id'):
                                        actual_component_order.append(name)
                            
                            print(f"🔧 Dynamic discovery: Found {len(actual_component_order)} valid components out of {len(component_names)} expected")
                            print(f"🔧 Received {len(args)} arguments from UI")
                            debug_print(f"🔧 Dynamic discovery: Found {len(actual_component_order)} valid components out of {len(component_names)} expected")
                            debug_print(f"🔧 Received {len(args)} arguments from UI")
                            
                            # Map each argument to its corresponding component name
                            # Use settings file values when available to bypass Gradio caching issues
                            for i, name in enumerate(actual_component_order):
                                if i < len(args):
                                    gradio_value = args[i]
                                    # Use settings file value if available and different from default
                                    if name in settings_data:
                                        settings_value = settings_data[name]
                                        # For critical parameters, prefer settings file over Gradio cache
                                        if name in ['W', 'H', 'strength', 'animation_prompts'] and settings_value != gradio_value:
                                            print(f"🔧 Using settings value for {name}: {settings_value} (instead of Gradio: {gradio_value})")
                                            kwargs[name] = settings_value
                                        else:
                                            kwargs[name] = gradio_value
                                    else:
                                        kwargs[name] = gradio_value
                                else:
                                    kwargs[name] = None
                            
                            # Debug critical mappings with actual discovery
                            print(f"🔍 Dynamic mapping results:")
                            critical_fields = ['strength', 'animation_prompts', 'W', 'H', 'mask_overlay_blur']
                            for field in critical_fields:
                                if field in kwargs:
                                    value = kwargs[field]
                                    print(f"   {field} = {str(value)[:50] if isinstance(value, str) else value}")
                                elif field in actual_component_order:
                                    pos = actual_component_order.index(field)
                                    print(f"   {field} at position {pos} (in actual order)")
                                else:
                                    print(f"   {field} = NOT FOUND in UI components")
                            
                            debug_print(f"🔍 Dynamic mapping results:")
                            for field in critical_fields:
                                if field in kwargs:
                                    value = kwargs[field]
                                    debug_print(f"   {field} = {str(value)[:50] if isinstance(value, str) else value}")
                                elif field in actual_component_order:
                                    pos = actual_component_order.index(field)
                                    debug_print(f"   {field} at position {pos} (in actual order)")
                                else:
                                    debug_print(f"   {field} = NOT FOUND in UI components")
                            
                            # Show first 10 actual mappings with detailed debug
                            print(f"🔍 First 10 dynamic mappings with args received:")
                            for i, name in enumerate(actual_component_order[:10]):
                                value = kwargs.get(name, 'None')
                                raw_arg = args[i] if i < len(args) else 'MISSING'
                                print(f"   {i:2d}: {name:20} = {str(value)[:30] if isinstance(value, str) else value} (raw: {str(raw_arg)[:20]})")
                            
                            # Show positions of critical components in actual discovery order
                            print(f"🔍 Critical component positions in ACTUAL discovery order:")
                            for field in critical_fields:
                                if field in actual_component_order:
                                    actual_pos = actual_component_order.index(field)
                                    expected_pos = component_names.index(field) if field in component_names else -1
                                    print(f"   {field:20} actual pos {actual_pos:3d}, expected pos {expected_pos:3d} (diff: {actual_pos - expected_pos:+3d})")
                            
                            debug_print(f"🔍 First 10 dynamic mappings:")
                            for i, name in enumerate(actual_component_order[:10]):
                                value = kwargs.get(name, 'None')
                                debug_print(f"   {i:2d}: {name:20} = {str(value)[:30] if isinstance(value, str) else value}")
                            
                            # Call the new named argument version
                            return run_deforum(**kwargs)
                        
                        return named_run_deforum
                    
                    # DYNAMIC COMPONENT COLLECTION: Only use components that actually exist
                    input_components = []
                    component_names = get_component_names()
                    actual_components = []
                    
                    print(f"🔧 Building input components using dynamic discovery...")
                    print(f"🔧 Available UI components: {len(components)} total")
                    debug_print(f"Building input components using dynamic discovery...")
                    debug_print(f"Available UI components: {len(components)} total")
                    
                    # Only collect components that actually exist and are valid
                    for name in component_names:
                        if name in components and components[name] is not None:
                            comp = components[name]
                            if hasattr(comp, '_id'):
                                input_components.append(comp)
                                actual_components.append(name)
                                debug_print(f"   ✅ {len(actual_components):3d}: {name}")
                    
                    debug_print(f"🔧 Collected {len(actual_components)} valid components out of {len(component_names)} expected")
                    
                    # Show which critical components are found
                    critical_fields = ['W', 'H', 'strength', 'animation_prompts', 'mask_overlay_blur']
                    print(f"🔍 Critical component availability:")
                    for field in critical_fields:
                        if field in actual_components:
                            pos = actual_components.index(field)
                            expected_pos = component_names.index(field) if field in component_names else -1
                            print(f"   ✅ {field:20} at actual pos {pos:3d}, expected pos {expected_pos:3d} (diff: {pos - expected_pos:+3d})")
                        else:
                            print(f"   ❌ {field:20} -> MISSING from UI")
                    
                    debug_print(f"🔍 Critical component availability:")
                    for field in critical_fields:
                        if field in actual_components:
                            pos = actual_components.index(field)
                            debug_print(f"   ✅ {field:20} at position {pos:3d}")
                        else:
                            debug_print(f"   ❌ {field:20} -> MISSING from UI")
                    
                    # Show first 10 actual components that will be sent to Gradio with their current values
                    print(f"🔍 First 10 components being sent to run_deforum with their CURRENT VALUES:")
                    for i, name in enumerate(actual_components[:10]):
                        comp = components.get(name)
                        try:
                            current_value = comp.value if hasattr(comp, 'value') else 'NO_VALUE'
                        except:
                            current_value = 'ERROR_GETTING_VALUE'
                        print(f"   {i:2d}: {name:20} = {str(current_value)[:40]}")
                    
                    debug_print(f"🔍 First 10 components being sent to run_deforum:")
                    for i, name in enumerate(actual_components[:10]):
                        debug_print(f"   {i:2d}: {name}")
                    
                    debug_print(f"Prepared {len(input_components)} input components in correct order")
                    
                    # Verify critical component positions
                    critical_positions = {}
                    for i, name in enumerate(component_names):
                        if name in ['W', 'H', 'strength', 'animation_prompts', 'mask_overlay_blur']:
                            critical_positions[name] = i
                    debug_print(f"Critical component positions: {critical_positions}")
                    
                    # Set up the generate button click handler with named argument wrapper
                    submit.click(
                        fn=wrap_gradio_gpu_call(create_named_run_wrapper(), extra_outputs=[None, '', '']),
                        inputs=input_components,
                        outputs=[
                            deforum_gallery,
                            generation_info,
                            html_info,
                            html_info
                        ],
                        show_progress=True
                    )
                    
                    debug_print("✅ Generate button connected with named arguments!")
                    
                except Exception as e:
                    print(f"⚠️ Failed to connect Generate button: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Connect settings buttons
                try:
                    if components:
                        # Create a simple list of all components for settings operations
                        all_components_list = []
                        for name in get_component_names():
                            if name in components and components[name] is not None:
                                all_components_list.append(components[name])
                        
                        if all_components_list:
                            # Save settings
                            save_settings_btn.click(
                                fn=save_settings,
                                inputs=[settings_path] + all_components_list,
                                outputs=[html_info]
                            )
                            
                            # Load all settings
                            load_settings_btn.click(
                                fn=load_all_settings,
                                inputs=[settings_path],
                                outputs=all_components_list + [html_info]
                            )
                            
                            # Load video settings
                            load_video_settings_btn.click(
                                fn=load_video_settings,
                                inputs=[settings_path],
                                outputs=all_components_list[:len(DeforumOutputArgs().keys())] + [html_info]
                            )
                            
                            debug_print("Settings buttons connected")
                        else:
                            debug_print("No components available for settings buttons")
                except Exception as e:
                    print(f"⚠️ Could not connect settings buttons: {e}")
                
                debug_print("Interface setup completed successfully")
                
            debug_print("Deforum interface created successfully!")
            return [(deforum_interface, "Deforum", "deforum_interface")]
        except Exception as e:
            import traceback
            print(f"⚠️ CRITICAL: Failed to create Deforum extension UI: {e}")
            print(f"⚠️ Full traceback:")
            print(traceback.format_exc())
            print(f"⚠️ Creating minimal fallback interface...")
            
            # Create a minimal fallback interface that should always work
            try:
                with gr.Blocks(analytics_enabled=False) as minimal_interface:
                    gr.HTML("<h2>⚠️ Deforum Extension Partially Loaded</h2>")
                    gr.HTML("<p>The extension encountered issues during full UI creation but is attempting to load with basic functionality.</p>")
                    gr.HTML(f"<p>Error: {str(e)}</p>")
                    
                return [(minimal_interface, "Deforum", "deforum_interface")]
            except Exception as fallback_error:
                print(f"⚠️ Even fallback interface failed: {fallback_error}")
                return []
    except Exception as e:
        print(f"⚠️ CRITICAL: Failed to create Deforum extension UI: {e}")
        print(f"⚠️ Creating minimal fallback interface...")
        
        # Create a minimal fallback interface that should always work
        try:
            with gr.Blocks(analytics_enabled=False) as minimal_interface:
                gr.HTML("<h2>⚠️ Deforum Extension Partially Loaded</h2>")
                gr.HTML("<p>The extension encountered issues during full UI creation but is attempting to load with basic functionality.</p>")
                gr.HTML(f"<p>Error: {str(e)}</p>")
                
            return [(minimal_interface, "Deforum", "deforum_interface")]
        except Exception as fallback_error:
            print(f"⚠️ Even fallback interface failed: {fallback_error}")
            return []