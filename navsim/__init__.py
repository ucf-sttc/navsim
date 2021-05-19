import os

# Path(__file__).parent.joinpath('version.txt')
with open(os.path.join(os.path.dirname(__file__), 'version.txt'), 'r') as vf:
    __version__ = vf.read().strip()
from .agents import DDPGAgent
from .memory import Memory
from .env import (
    NavSimGymEnv,
    navsimgymenv_creator,
)

from .executor import Executor