"""Make the dev tools in tools/ importable from tests.

They are deliberately not part of the installed package — scan_secrets.py runs in a
pre-commit hook and in CI, where importing the library would be beside the point.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
