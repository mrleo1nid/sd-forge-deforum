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
