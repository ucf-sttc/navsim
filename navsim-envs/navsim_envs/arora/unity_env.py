#ARORA

from abc import abstractmethod, abstractclassmethod
from importlib.resources import path
from pathlib import Path

import numpy as np
import time
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityWorkerInUseException

from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
from .side_channels import (
    MapSideChannel,
    NavigableSideChannel,
    ShortestPathSideChannel,
    SetAgentPositionSideChannel
)
#from .set_agent_position_side_channel import SetAgentPositionSideChannel
#from .shortest_path_side_channel import ShortestPathSideChannel
from .configs import default_env_config
from navsim_envs.base_envs import AroraUnityEnvBase

class AroraUnityEnv(AroraUnityEnvBase):
    """AroraUnityEnv Class is a wrapper to UnityEnvironment

    Read the **NavSim Environment Tutorial** on how to use this class.
    """
    #logger = logger
    actions = {
        'forward_left' : [1,-1,0],
        'forward_right' :[1, 1,0],
        'forward' : [1,0,0],
        'backward' : [-1,0,0]
    }

    def __init__(self, env_config) -> None:
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

        eng_sc = EngineConfigurationChannel()
        eng_sc.set_configuration_parameters(time_scale=0.25, quality_level=0)

        env_pc = EnvironmentParametersChannel()
        env_sfp = env_pc.set_float_parameter

        timeout = env_config['timeout'] + (0.5 * (env_config['start_from_episode'] - 1))

        ad_args = [
            "-allowedArea", f"{env_config['area']}",
            "-agentCarPhysics", f"{env_config['agent_car_physics']}",
            "-episodeLength", f"{env_config['episode_max_steps']}",
            "-fastForward", f"{env_config['start_from_episode'] - 1}",
            "-force-device-index", f"{env_config['env_gpu_id']}",
            "-force-vulkan" if (env_config["env_gpu_id"] > 0) else "",
            "-goalDistance", f"{env_config['goal_distance']}",
            "-goalClearance", f"{env_config['goal_clearance']}",
            "-goalSelectionIndex", f"{env_config['goal']}",
            "-numberOfTrafficVehicles", f"{env_config['traffic_vehicles']}",
            "-observationMode", f"{env_config['obs_mode']}",
            "-observationWidth", f"{env_config['obs_width']}",
            "-observationHeight", f"{env_config['obs_height']}",
            "-relativeSteering", f"{env_config['relative_steering']}",
            "-rewardForGoal", f"{env_config['reward_for_goal']}",
            "-rewardForNoViablePath", f"{env_config['reward_for_no_viable_path']}",
            "-rewardStepMul", f"{env_config['reward_step_mul']}",
            "-rewardCollisionMul", f"{env_config['reward_collision_mul']}",
            "-rewardSplDeltaMul", f"{env_config['reward_spl_delta_mul']}",
            "-showVisualObservations" if env_config['show_visual'] else "",
            "-saveStepLog" if env_config["debug"] else "",
            "-segmentationMode", f"{env_config['segmentation_mode']}",
            "-selectedTaskIndex", f"{env_config['task']}"
        ]

        # super().__init__(file_name=env_config['env_path'],
        #                 log_folder=env_config["log_folder"],
        #                 no_graphics=False,
        #                 seed=env_config["seed"],
        #                 timeout_wait=timeout,
        #                 worker_id=env_config["worker_id"],
        #                 base_port=env_config["base_port"],
        #                 side_channels=[eng_sc, env_pc,
        #                                self.map_side_channel,
        #                                self.fpc, self.nsc,
        #                                self.sapsc, self.spsc],
        #                 additional_args=ad_args)

        self._navsim_base_port = env_config['base_port']
        if self._navsim_base_port is None:
            self._navsim_base_port = UnityEnvironment.BASE_ENVIRONMENT_PORT if env_config[
                'env_path'] else UnityEnvironment.DEFAULT_EDITOR_PORT
        self._navsim_worker_id = env_config['worker_id']

        while True:
            try:
                env_config["worker_id"] = self._navsim_worker_id
                env_config["base_port"] = self._navsim_base_port
                # self.uenv = RideUnityEnv(env_config=env_config)
                super().__init__(file_name=env_config['env_path'],
                                 log_folder=env_config["log_folder"],
                                 no_graphics=False,
                                 seed=env_config["seed"],
                                 timeout_wait=timeout,
                                 worker_id=env_config["worker_id"],
                                 base_port=env_config["base_port"],
                                 side_channels=[eng_sc, env_pc,
                                                self.map_side_channel,
                                                self.fpc, self.nsc,
                                                self.sapsc, self.spsc
                                                ],
                                 additional_args=ad_args)
            except UnityWorkerInUseException:
                time.sleep(2)
                self._navsim_base_port += 1
            else:
                from_str = "Editor" if env_config['env_path'] is None else f"from {env_config['env_path']}"
                AroraUnityEnv.logger.info(f"Created UnityEnvironment {from_str} "
                                         f"at port {self._navsim_base_port + self._navsim_worker_id} "
                                         f"to start from episode {env_config['start_from_episode']}")
                break

        self.env_config = env_config

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
        self.process_immediate_message(self.spsc.build_immediate_request())
        #print('inside unity_env code:',self.spsc.path)
        return self.spsc.path

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

    def reset(self):
        UnityEnvironment.reset(self)
        self.map_side_channel.unity_max_x = 3284.0 
        self.map_side_channel.unity_max_z = 2666.3 
        #self.map_side_channel.unity_max_x = self.fpc.get_property("TerrainX") #3284.0 #
        #self.map_side_channel.unity_max_z = self.fpc.get_property("TerrainZ") #2666.3 #
        self.map_side_channel.navmap_max_x = int(self.map_side_channel.unity_max_x)
        self.map_side_channel.navmap_max_y = int(self.map_side_channel.unity_max_z)