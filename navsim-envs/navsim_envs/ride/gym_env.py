#RIDE

import numpy as np

from typing import Any, List, Union, Optional


from .configs import default_env_config

# @attr.s(auto_attribs=True)
# class AgentState:
#    position: Optional["np.ndarray"]
#    rotation: Optional["np.ndarray"] = None

from .unity_env import RideUnityEnv

from navsim_envs.envs_base import AroraGymEnvBase

def ridegymenv_creator(env_config):
    return RideGymEnv(env_config)  # return an env instance

class RideGymEnv(AroraGymEnvBase):
    """RideGymEnv inherits from Unity2Gym that inherits from the Gym interface.

    Read the **NavSim Environment Tutorial** on how to use this class.
    """

    def __init__(self, env_config) -> None:
        """
        env_config: The environment configuration dictionary Object
        """

        super().__init__(env_config, default_env_config, RideUnityEnv)
        if self.env_config['obs_mode']==1:
            self.metadata['render.modes']+=['rgb_array']

        # TODO: convert env_config to self.env_config so we can add missing values
        #   and use self.env_config to print in the info section
        # filename: Optional[str] = None, obs_mode: int = 0, max_steps:int = 5):
        """
        for key in default_env_config:
            if key not in env_config:
                env_config[key] = env_config

        log_folder = Path(env_config['log_folder']).resolve()
        # self.obs_mode = int(self.env_config.get('obs_mode', 2))
        if env_config['debug']:
            self.logger.setLevel(10)
        else:
            self.logger.setLevel(20)

        self._obs = None

        if env_config['obs_mode'] == 0:
            env_config["save_visual_obs"] = False
        #elif env_config['obs_mode'] == 1:
        #    env_config["save_vector_obs"] = False

        self.e_num = 0
        self.s_num = 0

        self._agent_position = None
        self._agent_velocity = None
        self._agent_rotation = None
        self._goal_position = None

        self.spl_start = self.spl_current = None

        #self._navsim_base_port = env_config['base_port']
        #if self._navsim_base_port is None:
        #    self._navsim_base_port = UnityEnvironment.BASE_ENVIRONMENT_PORT if env_config[
        #        'env_path'] else UnityEnvironment.DEFAULT_EDITOR_PORT
        #self._navsim_worker_id = env_config['worker_id']

        #while True:
        #    try:
        #        env_config["worker_id"] = self._navsim_worker_id
        #        env_config["base_port"] = self._navsim_base_port
        #        self.uenv = RideUnityEnv(env_config=env_config)
        self.uenv = RideUnityEnv(env_config=env_config)
        #    except UnityWorkerInUseException:
        #        time.sleep(2)
        #        self._navsim_base_port += 1
        #    else:
        #        from_str = "" if env_config['env_path'] is None else f"from {env_config['env_path']}"
        #        RideGymEnv.logger.info(f"Created UnityEnvironment {from_str} "
        #                                f"at port {self._navsim_base_port + self._navsim_worker_id} "
        #                                f"to start from episode {env_config['start_from_episode']}")
        #        break

        super().__init__(unity_env=self.uenv,
                         uint8_visual=False,
                         flatten_branched=False,
                         allow_multiple_obs=True,
                         action_space_seed=env_config['seed']
                         )

        #self._navigable_map = self.uenv.get_navigable_map()
        # TODO: the filenames should be prefixed with specific id of this instance of env

        # TODO: Read the file upto start_episode and purge the records
        self.actions_file = log_folder / 'actions.csv'
        if env_config['save_actions']:
            if (env_config["start_from_episode"] == 1) or (
                    self.actions_file.exists() == False):
                self.actions_file = self.actions_file.open(mode='w')
            else:
                self.actions_file = self.actions_file.open(mode='a')
            self.actions_writer = csv.writer(self.actions_file,
                                             delimiter=',',
                                             quotechar='"',
                                             quoting=csv.QUOTE_MINIMAL)

        if env_config['save_visual_obs'] and (env_config["obs_mode"] in [1]):
            self.rgb_folder = log_folder / 'rgb_obs'
            self.rgb_folder.mkdir(parents=True, exist_ok=True)
            self.dep_folder = log_folder / 'dep_obs'
            self.dep_folder.mkdir(parents=True, exist_ok=True)
            self.seg_folder = log_folder / 'seg_obs'
            self.seg_folder.mkdir(parents=True, exist_ok=True)
        else:
            env_config['save_visual_obs'] = False

        if env_config['save_vector_obs'] and (env_config["obs_mode"] in [0, 1]):
            self.sp_folder = log_folder / 'sp_obs'
            self.sp_folder.mkdir(parents=True, exist_ok=True)

            self.vec_file = log_folder / 'vec_obs.csv'
            if (env_config['start_from_episode'] == 1) or (
                    self.vec_file.exists() == False):
                self.vec_file = self.vec_file.open(mode='w')
                self.vec_writer = csv.writer(self.vec_file,
                                             delimiter=',',
                                             quotechar='"',
                                             quoting=csv.QUOTE_MINIMAL)
                self.vec_writer.writerow(
                    ['e_num', 's_num', 'spl_current', 'timestamp'] +
                    ['posx', 'posy', 'posz', 'velx', 'vely', 'velz',
                     'rotx', 'roty', 'rotz', 'rotw', 'goalx', 'goaly', 'goalz',
                     'proxforward', 'prox45left', 'prox45right'])
            else:
                # TODO: Read the file upto start_episode and purge the records
                self.vec_file = self.vec_file.open(mode='a')
                self.vec_writer = csv.writer(self.vec_file,
                                             delimiter=',',
                                             quotechar='"',
                                             quoting=csv.QUOTE_MINIMAL)
            self.vec_file.flush()
        else:
            env_config['save_vector_obs'] = False

        self.env_config = env_config
        """



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

        env_id = 'ride-v0'
        from gym.envs.registration import register, registry

        #env_dict = registry.env_specs.copy()
        #for env in env_dict:
        if env_id in registry.env_specs:
            print(f"navsim_envs: Removing {env_id} from Gym registry")
            del registry.env_specs[env_id]

        print(f"navsim_envs: Adding {env_id} to Gym registry")
        register(id=env_id, entry_point='navsim_envs.ride:RideGymEnv')

    @staticmethod
    def register_with_ray():
        """Registers the environment with ray registry with the name navsim

        """
        from ray.tune.registry import register_env
        register_env("ride-v0", ridegymenv_creator)



    