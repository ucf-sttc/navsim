
from abc import abstractmethod
import csv
from pathlib import Path
import time
from typing import Any, List, Union
from mlagents_envs.environment import UnityEnvironment
import numpy as np

from .envs_side_channels import (
    MapSideChannel,
    NavigableSideChannel,
    ShortestPathSideChannel,
    SetAgentPositionSideChannel
)
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel

from navsim_envs.util import (
    imwrite,
    logger
)

from gym_unity.envs import (
    UnityToGymWrapper,
    GymStepResult
)

class UnityEnvBase(UnityEnvironment):

    logger = logger

    @property
    @abstractmethod
    def unity_max_x(self) -> float:
        pass

    @property
    @abstractmethod
    def unity_max_z(self) -> float:
        pass

class AroraUnityEnvBase(UnityEnvBase):
    observation_modes = [0,1]

    def __init__(self, env_config, default_env_config) -> None:
        """
        env_config: The environment configuration dictionary Object
        """

        for key in default_env_config:
            if key not in env_config:
                env_config[key] = env_config

        log_folder = Path(env_config['log_folder']).resolve()
        log_folder.mkdir(parents=True, exist_ok=True)

        self.map_side_channel = MapSideChannel()
        self.fpc = FloatPropertiesChannel()
        self.nsc = NavigableSideChannel()
        self.sapsc = SetAgentPositionSideChannel()
        self.spsc = ShortestPathSideChannel()

        if env_config["env_path"] is not None:
            env_config["env_path"] = Path(env_config["env_path"]).resolve().as_posix()

    @property
    def navmap_max_x(self):
        return self.map_side_channel.navmap_max_x

    @property
    def navmap_max_y(self):
        return self.map_side_channel.navmap_max_y

    @property
    def unity_max_x(self) -> float:
        return self.map_side_channel.unity_max_x
        #return self.fpc.get_property("TerrainX")

    @property
    def unity_max_z(self) -> float:
        return self.map_side_channel.unity_max_z
        #return self.fpc.get_property("TerrainZ")

    @property
    def shortest_path_length(self):
        """the shortest navigable path length from current location to
        goal position
        """
        return self.fpc.get_property("ShortestPath")
    
    @property
    def shortest_path(self):
        self.process_immediate_message(
            self.spsc.build_immediate_request())
        return self.spsc.path

    def reset(self):
        super().reset()
        print("I am in reset")
        #self.map_side_channel.unity_max_x = 3284.0 
        #self.map_side_channel.unity_max_z = 2666.3
        print("fpc:",self.fpc.get_property("TerrainX"),self.fpc.get_property("TerrainZ")) 
        self.map_side_channel.unity_max_x = self.fpc.get_property("TerrainX") #3284.0 #
        self.map_side_channel.unity_max_z = self.fpc.get_property("TerrainZ") #2666.3 #
        self.map_side_channel.navmap_max_x = int(self.map_side_channel.unity_max_x)
        self.map_side_channel.navmap_max_y = int(self.map_side_channel.unity_max_z)

    def get_navigable_map(self) -> np.ndarray:
        """Get the Navigable Areas map

        Returns:
            A numpy array having 0 for non-navigable and 1 for navigable cells.

        Note:
            Current resolution is 3284 x 2666
        """

        self.process_immediate_message(
            self.map_side_channel.build_immediate_request("binaryMap"))

        return self.map_side_channel.requested_map

    def get_navigable_map_zoom(self, x: int, y: int) -> np.ndarray:
        """Get the Zoom into a cell in Navigable Areas map

        Returns:
            Zoomed in row, col location, a numpy array having 0 for non-navigable and 1 for navigable cells.

        """
        self.process_immediate_message(
            self.map_side_channel.build_immediate_request("binaryMapZoom", [y, x]))

        return self.map_side_channel.requested_map

    def get_navigable_map_zoom_area(self, x1: int, y1: int, x2:int, y2: int) -> np.ndarray:
        """Get the Zoom into a rectangle of cells bounded by x1,y1 - x2,y2 in Navigable Areas map

        Returns:
            Zoomed in row, col location, a numpy array having 0 for non-navigable and 1 for navigable cells.

        """

        if (x1 > x2):
            x1,x2 = x2,x1
        if (y1 > y2):
            y1,y2 = y2,y1

        area_list = []
        for y in range(y1,y2+1):
            row_list = []
            for x in range(x1,x2+1):
                row_list.append(self.get_navigable_map_zoom(x,y))    
            row = np.concatenate(row_list,axis=1)
            area_list.append(row)
        area = np.concatenate(area_list,axis=0)

        return area

    def sample_navigable_point(self, x: float = None, y: float = None,
                               z: float = None):
        """Provides a random sample of navigable point

        Args:
            x: x in unity's coordinate system
            y: y in unity's coordinate system
            z: z in unity's coordinate system

        Returns:
            If x,y,z are None, returns a randomly sampled navigable point [x,y,z].

            If x,z is given and y is None, returns True if x,z is navigable at some ht y else returns False.

            If x,y,z are given, returns True if x,y,z is navigable else returns False.
        """

        # if self.map_side_channel.requested_map is None:
        #    self.get_navigable_map(resolution_x, resolution_y,
        #                           cell_occupancy_threshold)
        #
        # idx = np.argwhere(self.map_side_channel.requested_map == 1)
        # idx_sample = np.random.choice(len(idx), replace=False)
        # return idx[idx_sample]
        if x is None or z is None:
            point = []
        else:
            if y is None:
                point = [x, z]
            else:
                point = [x, y, z]

        self.process_immediate_message(
            self.nsc.build_immediate_request("navigable", point))

        return self.nsc.point

class AroraGymEnvBase(UnityToGymWrapper):
    """AroraGymEnvBase inherits from Unity2Gym that inherits from the Gym interface.
    """

    logger = logger

        
    def __init__(self, env_config, default_env_config, uenv_class) -> None:
        """
        env_config: The environment configuration dictionary Object
        default_env_config: The default env_config for the environment that inherits from this
        """

        # TODO: convert env_config to self.env_config so we can add missing values
        #   and use self.env_config to print in the info section

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
        
        self.uenv = uenv_class(env_config=env_config)
        #    except UnityWorkerInUseException:
        #        time.sleep(2)
        #        self._navsim_base_port += 1
        #    else:
        #        from_str = "" if env_config['env_path'] is None else f"from {env_config['env_path']}"
        #        AroraGymEnv.logger.info(f"Created UnityEnvironment {from_str} "
        #                                f"at port {self._navsim_base_port + self._navsim_worker_id} "
        #                                f"to start from episode {env_config['start_from_episode']}")
        #        break

        super().__init__(unity_env=self.uenv,
                         uint8_visual=False,
                         flatten_branched=False,
                         allow_multiple_obs=True,
                         action_space_seed=env_config['seed']
                         )

        self._navigable_map = self.uenv.get_navigable_map()
        np.save(log_folder / 'navigable_map.npy',self._navigable_map)
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
            #self.sp_file = log_folder / 'sp_obs.csv'
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

    def _save_obs(self, obs):
        """Private method to save the observations in file

        Args:
            obs: observations object as returned by step() or reset()

        Returns:

        """
        if self.env_config['save_vector_obs']:
            self.vec_writer.writerow(
                [self.e_num, self.s_num, self.spl_current, time.time()] +
                list(obs[-1]))
            self.vec_file.flush()

            shortest_path = self.shortest_path
            if shortest_path is not None:
                filename = f'{self.e_num}_{self.s_num}.npy'
                np.save( self.sp_folder / filename,shortest_path)

        if self.env_config['save_visual_obs']:
            filename = f'{self.e_num}_{self.s_num}.jpg'
            imwrite(str(self.rgb_folder / filename), obs[0] * 255.0)
            if len(obs)>2:
                imwrite(str(self.dep_folder / filename), obs[1] * 255.0)
                imwrite(str(self.seg_folder / filename), obs[2] * 255.0)

    def _save_navigable_map_zoom(self):
        pass

    def _set_obs(self, s_):
        self._obs = s_
        if self.env_config['obs_mode'] in [0, 1]:
            vec_obs = list(self._obs[-1])
            self._agent_position = vec_obs[0:3]
            self._agent_velocity = vec_obs[3:6]
            self._agent_rotation = vec_obs[6:10]
            self._goal_position = vec_obs[10:13]

    def reset(self) -> Union[List[np.ndarray], np.ndarray]:
        s0 = super().reset()
        self.s_num = 0
        self.e_num += 1 if self.e_num else self.env_config['start_from_episode']
        self.spl_start = self.spl_current = self.shortest_path_length
        #if hasattr(self,'shortest_path_length'):
        #    self.spl_start = self.spl_current = self.shortest_path_length
        #else:
        #    self.spl_start = self.spl_current = None
        self._set_obs(s0)
        self._save_obs(self._obs)
        return s0

    def step(self, action: List[Any]) -> GymStepResult:
        s_, r, episode_done, info = super().step(action)
        self.s_num += 1
        self.spl_current = self.shortest_path_length
        #if hasattr(self,'shortest_path_length'):
        #    self.spl_current = self.shortest_path_length
        self._set_obs(s_)
        self._save_obs(self._obs)
        if self.env_config['save_actions']:
            self.actions_writer.writerow(
                [self.e_num, self.s_num] + list(action))
            self.actions_file.flush()
        return s_, r, episode_done, info

    
    # expose properties from uenv
    @property
    def actions(self):
        return self.uenv.actions

    @property
    def shortest_path(self):
        return self.uenv.shortest_path
    
    @property
    def shortest_path_length(self):
        """the shortest navigable path length from current location to
        goal position
        """
        return self.uenv.shortest_path_length

    # expose methods from uenv
    def get_navigable_map(self) -> np.ndarray:
        """Get the Navigable Areas map

        Returns:
            A numpy array having 0 for non-navigable and 1 for navigable cells.

        Note:
            Current resolution is 3284 x 2666
        """

        return self.uenv.get_navigable_map()

    def get_navigable_map_zoom(self, x: int, y: int) -> np.ndarray:
        """Get the Navigable Areas map

        Returns:
            Zoomed in row, col location, a numpy array having 0 for non-navigable and 1 for navigable cells.

        """
        return self.uenv.get_navigable_map_zoom(x=x, y=y)

    
    def get_navigable_map_zoom_area(self, x1: int, y1: int, x2:int, y2: int) -> np.ndarray:
        """Get the Zoom into a rectangle of cells bounded by x1,y1 - x2,y2 in Navigable Areas map

        Returns:
            Zoomed in row, col location, a numpy array having 0 for non-navigable and 1 for navigable cells.

        """
        return self.uenv.get_navigable_map_zoom_area(x1=x1,x2=x2,y1=y1,y2=y2)

    # Functions added to have parity with Env and RLEnv of habitat lab

    @property
    def agent_obs(self):
        """Agent observations
        """
        if self._obs is None:
            raise EnvNotInitializedError()
        return self._obs

    @property
    def agent_position(self):
        """Position of agent in unity coordinates x,y,z
        """
        if self._agent_position is None:
            raise EnvNotInitializedError()

        return self._agent_position

    @property
    def agent_rotation(self):
        """Rotation of agent in unity quaternions x,y,z,w
        """
        if self._agent_rotation is None:
            raise EnvNotInitializedError()

        return self._agent_rotation

    @property
    def agent_velocity(self):
        """Velocity of agent in unity coordinates x,y,z
        """
        if self._agent_velocity is None:
            raise EnvNotInitializedError()
        return self._agent_velocity

    # sim.get_agent_state() -> agent_x, y, orientation
    # sim.set_agent_state(position, orientation)
    # sim.get_observations_at(position, orientation) -> observation when agent is at position with specified orientation
    # sim.sample_navigable_point() -> agent_x,y (must be a navigable location in the map)

    @property
    def goal_position(self):
        """Position of goal in unity coordinates x,y,z
        """
        if self._goal_position is None:
            raise EnvNotInitializedError()

        return self._goal_position

    @property
    def current_episode_num(self):
        """Currently executing episode number, 0 means env just initialized
        """
        return self.e_num

    @property
    def last_step_num(self):
        """Last executed step number, 0 mean env just initialized or reset
        """
        return self.s_num


