import os

# Path(__file__).parent.joinpath('version.txt')
with open(os.path.join(os.path.dirname(__file__), 'version.txt'), 'r') as vf:
    __version__ = vf.read().strip()
from navsim.agent.agents import DDPGAgent
from .memory import (
    CupyMemory,
    NumpyMemory,
)

from .env import (
    NavSimGymEnv,
    navsimgymenv_creator,
)

NavSimGymEnv.register_with_gym()

from .executor import Executor
