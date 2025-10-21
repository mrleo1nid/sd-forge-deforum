/*
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

Contact the authors: https://deforum.github.io/
*/

function submit_deforum(){
    rememberGallerySelection('deforum_gallery')
    showSubmitButtons('deforum', false)

    var id = randomId()
    requestProgress(id, gradioApp().getElementById('deforum_gallery_container'), gradioApp().getElementById('deforum_gallery'), function(){
        showSubmitButtons('deforum', true)
    })

    var res = create_submit_args(arguments)

    res[0] = id

    return res
}

// UI Customization: Apply slopcore gradient and hide unwanted buttons
function applyDeforumUICustomization() {
    console.log('🎨 Applying Deforum UI customization...');

    // Apply slopcore gradient to Generate button
    const generateSelectors = [
        '#deforum_generate',
        '#deforum_generate button',
        'button#deforum_generate',
        '[id*="deforum_generate"] button'
    ];

    let generateBtn = null;
    for (const selector of generateSelectors) {
        generateBtn = gradioApp().querySelector(selector);
        if (generateBtn) {
            console.log('✓ Found Generate button with selector:', selector);
            generateBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            generateBtn.style.border = 'none';
            generateBtn.style.color = 'white';
            generateBtn.style.fontWeight = '600';
            generateBtn.style.textShadow = '0 1px 2px rgba(0,0,0,0.2)';
            generateBtn.style.boxShadow = '0 4px 6px rgba(102, 126, 234, 0.3)';
            generateBtn.style.transition = 'all 0.3s ease';
            break;
        }
    }

    if (!generateBtn) {
        console.log('❌ Generate button not found yet');
    }

    // Hide unwanted buttons in results area, keep only folder button
    const resultsDiv = gradioApp().querySelector('#deforum_results');
    if (resultsDiv) {
        const allButtons = resultsDiv.querySelectorAll('button');
        console.log(`Found ${allButtons.length} buttons in #deforum_results`);

        allButtons.forEach((btn, idx) => {
            const btnId = btn.id || 'no-id';
            const btnText = (btn.textContent || '').trim();
            console.log(`  Button ${idx}: id="${btnId}", text="${btnText}"`);

            // Keep only the folder button
            if (!btnId.includes('open_folder') && !btnText.includes('📁')) {
                btn.style.display = 'none';
                btn.style.visibility = 'hidden';
                btn.style.opacity = '0';
                btn.style.width = '0';
                btn.style.height = '0';
                btn.style.padding = '0';
                btn.style.margin = '0';
                console.log(`  ✓ Hidden button: ${btnId}`);
            }
        });
    } else {
        console.log('❌ #deforum_results not found yet');
    }
}

// Apply UI customization on load and when DOM updates
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎨 Deforum UI customization script loaded');

    // Initial application
    setTimeout(applyDeforumUICustomization, 500);
    setTimeout(applyDeforumUICustomization, 1500);
    setTimeout(applyDeforumUICustomization, 3000);

    // Re-apply when Gradio updates the DOM
    const observer = new MutationObserver(function() {
        applyDeforumUICustomization();
    });

    observer.observe(gradioApp(), { childList: true, subtree: true });
    console.log('👀 MutationObserver started for Deforum UI customization');
});