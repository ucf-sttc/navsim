# navsim_envs

from pathlib import Path

version_file = Path(__file__).parent.joinpath('version.txt')
if not version_file.exists():
    version_file = Path(__file__).parent.joinpath('../../version.txt')
with open(version_file, 'r') as vf:
    __version__ = vf.read().strip()

__version_banner__ = f'=========================================\n' \
                     f'navsim_env version {__version__}\n' \
                     f'=========================================\n'

from . import arora

__all__ = ['arora']
