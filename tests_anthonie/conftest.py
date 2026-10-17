"""Make the repo-root modules (slice, params, properties) importable under plain `pytest`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
