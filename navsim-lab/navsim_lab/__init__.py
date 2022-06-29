from pathlib import Path

version_file = Path(__file__).parent.joinpath('version.txt')
if not version_file.exists():
    version_file = Path(__file__).parent.joinpath('../../version.txt')
with open(version_file, 'r') as vf:
    __version__ = vf.read().strip()



from navsim_lab import util, agent, executor


__all__ = ['agent', 'util', 'executor']
