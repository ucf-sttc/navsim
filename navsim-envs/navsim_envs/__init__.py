# navsim_envs

import os

# Path(__file__).parent.joinpath('version.txt')
with open(os.path.join(os.path.dirname(__file__), 'version.txt'), 'r') as vf:
    __version__ = vf.read().strip()

__version_banner__ = f'=========================================\n' \
                     f'navsim_env version {__version__}\n' \
                     f'=========================================\n'

from . import env, util

__all__ = ['env', 'util']
