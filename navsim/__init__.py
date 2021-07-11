import os

# Path(__file__).parent.joinpath('version.txt')
with open(os.path.join(os.path.dirname(__file__), 'version.txt'), 'r') as vf:
    __version__ = vf.read().strip()

from navsim.env import NavSimGymEnv

NavSimGymEnv.register_with_gym()

from navsim import util, agent, executor

__all__ = ['agent', 'util', 'executor']
