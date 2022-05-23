#ARORA

import numpy as np

from typing import Any, List, Union, Optional

from .configs import default_env_config

# @attr.s(auto_attribs=True)
# class AgentState:
#    position: Optional["np.ndarray"]
#    rotation: Optional["np.ndarray"] = None

from .unity_env import AroraUnityEnv

from mlagents_envs.rpc_utils import steps_from_proto

from navsim_envs.envs_base import AroraGymEnvBase

def navsimgymenv_creator(env_config):
    return AroraGymEnv(env_config)  # return an env instance

class AroraGymEnv(AroraGymEnvBase):
    """AroraGymEnv inherits from Unity2Gym that inherits from the Gym interface.

    Read the **NavSim Environment Tutorial** on how to use this class.
    """

    def __init__(self, env_config) -> None:
        """
        env_config: The environment configuration dictionary Object
        """
        super().__init__(env_config, default_env_config, AroraUnityEnv)
        if self.env_config['obs_mode']==1:
            self.metadata['render.modes']+=['rgb_array', 'depth', 'segmentation']


    # def close(self):
    #    if self.save_vector_obs:
    #        self.vec_file.close()
    #    if self.save_vector_obs or self.save_visual_obs:
    #        self.actions_file.close()
    #    super().close()

    #    @property
    #    def observation_space_shapes(self) -> list:
    #        """Returns the dimensions of the observation space
    #        """
    #        return [obs.shape for obs in self.observation_space.spaces]
    #
    #    @property
    #    def observation_space_types(self) -> list:
    #        """Returns the dimensions of the observation space
    #        """
    #        return [type(obs) for obs in self.observation_space.spaces]

    @staticmethod
    def register_with_gym():
        """Registers the environment with gym registry with the name navsim

        """

        env_id = 'arora-v0'
        from gym.envs.registration import register, registry

        #env_dict = registry.env_specs
        #if gym.envs.spec(env_id).id == env_id:
        #    print(f"navsim_envs: Removing {env} from Gym registry")
        #    del registry.env_specs[env]

        #for env in registry.env_specs:
        #    if env_id in env:
        #        print(f"navsim_envs: Removing {env} from Gym registry")
        #        del registry.env_specs[env]

        print(f"navsim_envs: Adding {env_id} to Gym registry")
        register(id=env_id, entry_point='navsim_envs.arora:AroraGymEnv')

    @staticmethod
    def register_with_ray():
        """Registers the environment with ray registry with the name navsim

        """
        from ray.tune.registry import register_env
        register_env("arora-v0", navsimgymenv_creator)










