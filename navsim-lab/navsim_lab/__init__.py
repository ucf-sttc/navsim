from pathlib import Path
from navsim_lab import util, agent, executor

version_file = Path(__file__).parent.joinpath("version.txt")
if not version_file.exists():
    version_file = Path(__file__).parent.joinpath("../../version.txt")
with open(version_file, "r") as vf:
    __version__ = vf.read().strip()


__all__ = ["agent", "util", "executor"]
