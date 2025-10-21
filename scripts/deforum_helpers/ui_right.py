# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

import os
from .args import DeforumOutputArgs, get_component_names, get_settings_component_names
from modules.shared import opts, state
from modules.ui import create_output_panel, wrap_gradio_call
from modules.call_queue import wrap_gradio_gpu_call
from .run_deforum import run_deforum
from .settings import save_settings, load_all_settings, load_video_settings, get_default_settings_path, update_settings_path
from .general_utils import get_deforum_version, get_commit_date
from .ui_left import setup_deforum_left_side_ui
from scripts.deforum_extend_paths import deforum_sys_extend
import gradio as gr

def on_ui_tabs():
    # extend paths using sys.path.extend so we can access all of our files and folders
    deforum_sys_extend()
    # set text above generate button
    style = '"text-align:center;font-weight:bold;margin-bottom:0em"'
    extension_url = "https://github.com/Tok/sd-forge-deforum"
    link = f"<a href='{extension_url}' target='_blank'>Zirteqs Fluxabled Fork</a>"
    extension_name = f"{link} of the Deforum Extension for WebUI Forge"

    commit_info = f"Git commit: {get_deforum_version()}"
    i1_store_backup = f"<p style={style}>{extension_name} - Version: {get_commit_date()} | {commit_info}</p>"
    i1_store = i1_store_backup

    # Slopcore gradient aesthetic for Generate button and hide unwanted buttons
    slopcore_css = """
    /* Slopcore gradient for Generate button - multiple selectors for Gradio 4 compatibility */
    #deforum_generate,
    button#deforum_generate,
    #deforum_generate > button,
    [id*="deforum_generate"] button,
    .generate-box-generating,
    .generate-box-interrupting {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    #deforum_generate:hover,
    button#deforum_generate:hover,
    #deforum_generate > button:hover,
    [id*="deforum_generate"] button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Hide ALL unwanted buttons in deforum results - keep only folder button */
    #deforum_results button[id*="save"],
    #deforum_results button[id*="send"],
    #save_deforum,
    #save_zip_deforum,
    #deforum_send_to_img2img,
    #deforum_send_to_inpaint,
    #deforum_send_to_extras,
    [id^="save_"][id$="_deforum"],
    [id^="deforum_send_"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Golden ratio depth preview scaling */
    #deforum_depth_gallery {
        transform: scale(0.618) !important;
        transform-origin: top left !important;
        margin-bottom: -20% !important;
    }
    """

    with gr.Blocks(analytics_enabled=False, css=slopcore_css) as deforum_interface:
        # Inject JavaScript for runtime UI customization
        gr.HTML(value="""
        <script>
        (function() {
            console.log('🎨 Deforum UI customization script loaded');

            function applyDeforumStyles() {
                console.log('🎨 Applying Deforum custom styles...');

                // Find all buttons in deforum_results and log them
                const resultsDiv = document.querySelector('#deforum_results');
                if (resultsDiv) {
                    console.log('Found #deforum_results');
                    const allButtons = resultsDiv.querySelectorAll('button');
                    console.log(`Found ${allButtons.length} buttons in results area`);

                    allButtons.forEach((btn, idx) => {
                        const btnId = btn.id || 'no-id';
                        const btnText = btn.textContent || 'no-text';
                        console.log(`  Button ${idx}: id="${btnId}", text="${btnText}"`);

                        // Hide everything except folder button
                        if (!btnId.includes('open_folder')) {
                            btn.style.display = 'none';
                            console.log(`  ✓ Hidden button: ${btnId}`);
                        }
                    });
                } else {
                    console.log('❌ #deforum_results not found yet');
                }

                // Apply slopcore gradient to Generate button
                const generateBtn = document.querySelector('#deforum_generate button') ||
                                   document.querySelector('button#deforum_generate');

                if (generateBtn) {
                    console.log('✓ Found Generate button, applying gradient...');
                    generateBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
                    generateBtn.style.border = 'none';
                    generateBtn.style.color = 'white';
                    generateBtn.style.fontWeight = '600';
                    generateBtn.style.textShadow = '0 1px 2px rgba(0,0,0,0.2)';
                    generateBtn.style.boxShadow = '0 4px 6px rgba(102, 126, 234, 0.3)';
                } else {
                    console.log('❌ Generate button not found yet');
                }
            }

            // Apply on load with multiple attempts
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', applyDeforumStyles);
            } else {
                applyDeforumStyles();
            }

            setTimeout(applyDeforumStyles, 500);
            setTimeout(applyDeforumStyles, 1500);
            setTimeout(applyDeforumStyles, 3000);

            // Re-apply when Gradio updates the DOM
            const observer = new MutationObserver(() => {
                applyDeforumStyles();
            });

            setTimeout(() => {
                observer.observe(document.body, { childList: true, subtree: true });
                console.log('👀 MutationObserver started');
            }, 100);
        })();
        </script>
        """, visible=False)

        components = {}
        dummy_component = gr.Button(visible=False)
        with gr.Row(elem_id='deforum_progress_row', equal_height=False, variant='compact'):
            with gr.Column(scale=1, variant='panel'):
                # setting the left side of the ui:
                components = setup_deforum_left_side_ui()
            with gr.Column(scale=1, variant='compact'):
                with gr.Row(variant='compact'):
                    btn = gr.Button("Click here after the generation to show the video")
                    components['btn'] = btn
                    close_btn = gr.Button("Close the video", visible=False)
                with gr.Row(variant='compact'):
                    i1 = gr.HTML(i1_store, elem_id='deforum_header')
                    components['i1'] = i1
                    def show_vid(): # Show video button related func
                        from .run_deforum import last_vid_data # get latest vid preview data (this import needs to stay inside the function!)
                        return {
                            i1: gr.update(value=last_vid_data, visible=True),
                            close_btn: gr.update(visible=True),
                            btn: gr.update(value="Update the video", visible=True),
                        }
                    btn.click(
                        fn=show_vid,
                        inputs=[],
                        outputs=[i1, close_btn, btn],
                        )
                    def close_vid(): # Close video button related func
                        return {
                            i1: gr.update(value=i1_store_backup, visible=True),
                            close_btn: gr.update(visible=False),
                            btn: gr.update(value="Click here after the generation to show the video", visible=True),
                        }
                    
                    close_btn.click(
                        fn=close_vid,
                        inputs=[],
                        outputs=[i1, close_btn, btn],
                        )
                id_part = 'deforum'
                with gr.Row(elem_id=f"{id_part}_generate_box", variant='compact'):
                    skip = gr.Button('Pause/Resume', elem_id=f"{id_part}_skip", visible=False)
                    interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt", visible=True)
                    interrupting = gr.Button('Interrupting...', elem_id=f"{id_part}_interrupting", elem_classes="generate-box-interrupting", tooltip="Interrupting generation...")
                    submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')

                    skip.click(
                        fn=lambda: state.skip(),
                        inputs=[],
                        outputs=[],
                    )

                    interrupt.click(
                        fn=lambda: state.interrupt(),
                        inputs=[],
                        outputs=[],
                    )

                    interrupting.click(
                        fn=lambda: state.interrupt(),
                        inputs=[],
                        outputs=[],
                    )
                
                # Use Deforum-specific output directory
                deforum_outdir = os.path.join(os.getcwd(), 'outputs', 'deforum')
                os.makedirs(deforum_outdir, exist_ok=True)
                res = create_output_panel("deforum", deforum_outdir)

                #deforum_gallery, generation_info, html_info, _

                generation_info = res.generation_info
                html_info= res.html_log
                deforum_gallery = res.gallery

                # Depth Preview Gallery - shown when save_depth_maps is enabled and animation_mode is 3D
                # Start visible=True, will be hidden by update function if conditions not met
                with gr.Accordion("🗺️ Depth Map Preview", open=False, visible=True, elem_id="deforum_depth_preview_accordion") as depth_preview_accordion:
                    gr.Markdown("""
                    **Depth maps will be saved to:** `[output_dir]/depth-maps/`

                    After generation completes, depth maps can be found in the depth-maps subdirectory.
                    *Live preview during generation coming soon.*
                    """)
                    depth_gallery = gr.Gallery(
                        label="Depth Maps",
                        show_label=False,
                        elem_id="deforum_depth_gallery",
                        columns=4,
                        height="auto"
                    )

                components['depth_preview_accordion'] = depth_preview_accordion
                components['depth_gallery'] = depth_gallery

                with gr.Row(variant='compact'):
                    settings_path = gr.Textbox(get_default_settings_path(), elem_id='deforum_settings_path', label="Settings File", info="Settings are automatically loaded on startup. Path can be relative to webui folder OR full/absolute.")
                with gr.Row(variant='compact'):
                    save_settings_btn = gr.Button('Save Settings', elem_id='deforum_save_settings_btn')
                    load_settings_btn = gr.Button('Load All Settings', elem_id='deforum_load_settings_btn')
                    load_video_settings_btn = gr.Button('Load Video Settings', elem_id='deforum_load_video_settings_btn')

        component_list = [components[name] for name in get_component_names()]

        submit.click(
                    fn=wrap_gradio_gpu_call(run_deforum),
                    _js="submit_deforum",
                    inputs=[dummy_component, dummy_component] + component_list,
                    outputs=[
                         deforum_gallery,
                         components["resume_timestring"],
                         generation_info,
                         html_info                 
                    ],
                )
        
        settings_component_list = [components[name] for name in get_settings_component_names()]
        video_settings_component_list = [components[name] for name in list(DeforumOutputArgs().keys())]

        save_settings_btn.click(
            fn=wrap_gradio_call(save_settings),
            inputs=[settings_path] + settings_component_list + video_settings_component_list,
            outputs=[],
        )
        
        # Create a path update function
        def path_updating_load_settings(*args):
            path = args[0]
            settings_path.value = path
            return load_all_settings(*args)
            
        load_settings_btn.click(
            fn=wrap_gradio_call(path_updating_load_settings),
            inputs=[settings_path] + settings_component_list,
            outputs=settings_component_list,
        )

        # Create a path update function for video settings
        def path_updating_load_video_settings(*args):
            path = args[0]
            settings_path.value = path
            return load_video_settings(*args)
            
        load_video_settings_btn.click(
            fn=wrap_gradio_call(path_updating_load_video_settings),
            inputs=[settings_path] + video_settings_component_list,
            outputs=video_settings_component_list,
        )

        # Depth preview visibility toggle based on save_depth_maps and animation_mode
        def update_depth_preview_visibility(save_depth, anim_mode):
            # Show depth preview if depth maps are enabled and using 3D mode
            should_show = save_depth and anim_mode == '3D'
            return gr.update(visible=should_show, open=should_show)

        components['save_depth_maps'].change(
            fn=update_depth_preview_visibility,
            inputs=[components['save_depth_maps'], components['animation_mode']],
            outputs=[depth_preview_accordion]
        )

        components['animation_mode'].change(
            fn=update_depth_preview_visibility,
            inputs=[components['save_depth_maps'], components['animation_mode']],
            outputs=[depth_preview_accordion]
        )

        # Also update visibility when settings are loaded
        load_settings_btn.click(
            fn=update_depth_preview_visibility,
            inputs=[components['save_depth_maps'], components['animation_mode']],
            outputs=[depth_preview_accordion]
        )

        load_video_settings_btn.click(
            fn=update_depth_preview_visibility,
            inputs=[components['save_depth_maps'], components['animation_mode']],
            outputs=[depth_preview_accordion]
        )

    # handle settings loading on UI launch
    def trigger_load_general_settings():
        print("Loading general settings...")
        
        # First check if deforum_settings.txt exists in webui root
        import os
        from modules import paths_internal
        webui_root_settings = os.path.join(paths_internal.script_path, "deforum_settings.txt")
        
        # Determine the settings file to load
        if os.path.isfile(webui_root_settings):
            # Use the settings file from webui root if it exists
            settings_file_path = webui_root_settings
            print(f"Loading existing settings from webui root: {settings_file_path}")
        else:
            # Fall back to default settings provided by the extension
            settings_file_path = get_default_settings_path()
            print(f"No settings found in webui root, using default settings from: {settings_file_path}")
        
        # Update the settings path field with the path
        settings_path.value = settings_file_path
        
        # Now call load_all_settings with ui_launch=True to update all components
        wrapped_fn = wrap_gradio_call(lambda *args, **kwargs: load_all_settings(*args, ui_launch=True, **kwargs))
        inputs = [settings_file_path] + [component.value for component in settings_component_list]
        outputs = settings_component_list
        updated_values = wrapped_fn(*inputs, *outputs)[0]
        
        # Update all the component values
        settings_component_name_to_obj = {name: component for name, component in zip(get_settings_component_names(), settings_component_list)}
        for key, value in updated_values.items():
            if key in settings_component_name_to_obj:
                settings_component_name_to_obj[key].value = value['value']

        # Update depth preview visibility based on loaded settings
        save_depth = components['save_depth_maps'].value
        anim_mode = components['animation_mode'].value
        should_show = save_depth and anim_mode == '3D'
        depth_preview_accordion.visible = should_show
        depth_preview_accordion.open = should_show
        print(f"Depth preview accordion: visible={should_show} (save_depth={save_depth}, anim_mode={anim_mode})")

    # Always load settings on startup - either from persistent settings path (if enabled),
    # from webui root, or from the extension's default settings
    trigger_load_general_settings()
        
    return [(deforum_interface, "Deforum", "deforum_interface")]