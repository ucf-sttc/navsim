import os
with open(os.path.join(os.path.dirname(__file__),'version.txt'), 'r') as vf:
    __version__ = vf.read().strip()
#TODO remove these imports from here
from .agents import DDPGAgent
from .memory import Memory
from .env import NavSimEnv
from .executor import Executor


