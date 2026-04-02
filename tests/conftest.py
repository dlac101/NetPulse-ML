"""Test configuration and fixtures.

Note: Individual test modules import only what they need.
The full FastAPI app is only imported for integration tests.
"""

import sys
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
