"""Pytest configuration for Deforum tests.

This file is automatically loaded by pytest before running tests.
It configures the Python path so that the deforum package can be imported.
"""

import sys
from pathlib import Path

# Add the extension root directory to Python path
# This allows `from deforum.utils import ...` to work
extension_root = Path(__file__).parent.parent
sys.path.insert(0, str(extension_root))

# Add Forge root directory to Python path
# This allows `import modules` to work (Forge's modules package)
forge_root = extension_root.parent.parent
sys.path.insert(0, str(forge_root))

# Call deforum_sys_extend() to properly set up all paths
# This is required for Deforum to import properly
try:
    from scripts.deforum_extend_paths import deforum_sys_extend
    deforum_sys_extend()
except ImportError:
    # If we can't import it, the basic path setup above should work
    pass
