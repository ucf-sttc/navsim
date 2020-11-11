from pathlib import Path
import sys

#===== TODO: Remove this when ezai_util is an installed package
base_path=Path.home() / 'projects'

for pkg in ['ezai_util']:
    pkg_path = base_path / pkg
    pkg_path = str(pkg_path.resolve())
    print(pkg_path)
    if not pkg_path in sys.path:
        sys.path.append(pkg_path)
#===== TODO: Remove this when ezai_util is an installed package
import ezai_util

from .classes import DDPGAgent
from .memory import Memory
from .env import NavSimEnv
from .trainer import Trainer

