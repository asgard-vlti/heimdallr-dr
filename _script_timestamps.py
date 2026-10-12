"""Load the shared stdlib parser for direct scripts without installing the package."""

import importlib.util
from pathlib import Path


_path = Path(__file__).resolve().parent / "src" / "heimdallrdr" / "_timestamps.py"
_spec = importlib.util.spec_from_file_location("_heimdallrdr_timestamps", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
parse_time_seconds_and_str = _module.parse_time_seconds_and_str
