# navsim_envs

import os

# Path(__file__).parent.joinpath('version.txt')
with open(os.path.join(os.path.dirname(__file__), 'version.txt'), 'r') as vf:
    __version__ = vf.read().strip()

from navsim_envs import env, util

__all__ = ['env','util']