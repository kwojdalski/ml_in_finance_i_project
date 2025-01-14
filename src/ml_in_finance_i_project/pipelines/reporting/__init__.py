"""Complete reporting pipeline for the spaceflights tutorial"""

import sys
from pathlib import Path

from .pipeline import create_pipeline  # NOQA

path = Path(__file__).parent.parent.parent
print(path)
path = path / "src"
sys.path.append(str(path))
