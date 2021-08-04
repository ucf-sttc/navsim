import csv
import struct
from pathlib import Path
import numpy as np

from typing import Any, List, Union, Optional

import time
import math

from gym_unity.envs import (
    UnityToGymWrapper,
    GymStepResult
)

# @attr.s(auto_attribs=True)
# class AgentState:
#    position: Optional["np.ndarray"]
#    rotation: Optional["np.ndarray"] = None

from ..util.exceptions import EnvNotInitializedError

from mlagents_envs.logging_util import get_logger

from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.environment import UnityEnvironment

from mlagents_envs.side_channel.engine_configuration_channel import \
    EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import \
    EnvironmentParametersChannel

from mlagents_envs.side_channel.float_properties_channel import \
    FloatPropertiesChannel

from .map_side_channel import MapSideChannel
from .navigable_side_channel import NavigableSideChannel
from .set_agent_position_side_channel import SetAgentPositionSideChannel

try:
    from cv2 import imwrite as imwrite

    print("navsim_envs: using cv2 as image library")
except ImportError as error:
    try:
        from imageio import imwrite as imwrite

        print("navsim_envs: using imageio as image library")
    except ImportError as error:
        try:
            from matplotlib.pyplot import imsave as imwrite

            print("navsim_envs: using matplotlib as image library")
        except ImportError as error:
            try:
                from PIL import Image

                print("navsim_envs: using PIL as image library")


                def imwrite(filename, arr):
                    im = Image.fromarray(arr)
                    im.save(filename)
            except ImportError as error:
                def imwrite(filename=None, arr=None):
                    print("navsim_envs: unable to load any of the following "
                          "image libraries: cv2, imageio, matplotlib, "
                          "PIL. Install one of these libraries to "
                          "save visuals.")


                imwrite()


def navsimgymenv_creator(env_config):
    return AroraGymEnv(env_config)  # return an env instance


class AroraGymEnv(UnityToGymWrapper):
    """AroraGymEnv inherits from Unity2Gym that inherits from the Gym interface.

    Read the **NavSim Environment Tutorial** on how to use this class.
    """
    metadata = {
        'render.modes': ['rgb_array', 'depth', 'segmentation', 'vector']}
    logger = get_logger(__name__)

    def __init__(self, env_config) -> None:
        """
        env_config: The environment configuration dictionary Object
        """

        # TODO: convert env_config to self.env_config so we can add missing values
        #   and use self.env_config to print in the info section
        # filename: Optional[str] = None, obs_mode: int = 0, max_steps:int = 5):
        self.env_config = env_config
        self.obs_mode = int(self.env_config.get('obs_mode', 2))
        self.start_from_episode = int(
            self.env_config.get('start_from_episode', 1))
        self.debug = env_config.get("debug", False)
        if self.debug:
            self.logger.setLevel(10)
        else:
            self.logger.setLevel(20)

        self.obs = None

        if self.obs_mode == 0:
            env_config["save_visual_obs"] = False
        elif self.obs_mode == 1:
            env_config["save_vector_obs"] = False

        self.save_actions = env_config.get("save_actions", False)
        self.save_visual_obs = env_config.get("save_visual_obs", False)
        self.save_vector_obs = env_config.get("save_vector_obs", False)

        self.e_num = 0
        self.s_num = 0

        self._agent_position = None
        self._agent_velocity = None
        self._agent_rotation = None
        self._goal_position = None

        self.spl_start = self.spl_current = None
        self.reward_spl_delta_mul = float(
            self.env_config.get('reward_spl_delta_mul', 1))

        # self.run_base_folder_str = env_config.get("run_base_folder_str", '.')
        # self.run_base_folder = Path(self.run_base_folder_str)
        seed = int(self.env_config.get('seed') or 0)

        # if self._env:
        #    raise ValueError('Environment already open')
        # else:

        self.map_side_channel = MapSideChannel()
        self.fpc = FloatPropertiesChannel()
        self.nsc = NavigableSideChannel()
        self.sapsc = SetAgentPositionSideChannel()
        # print(self.map_side_channel
        # )

        eng_sc = EngineConfigurationChannel()
        eng_sc.set_configuration_parameters(time_scale=10, quality_level=0)

        env_pc = EnvironmentParametersChannel()
        env_sfp = env_pc.set_float_parameter

        env_sfp("rewardForGoal",
                float(self.env_config.get('reward_for_goal', 50)))
        env_sfp("rewardForNoViablePath",
                float(self.env_config.get('reward_for_no_viable_path', -50)))
        env_sfp("rewardStepMul",
                float(self.env_config.get('reward_step_mul', 0.1)))
        env_sfp("rewardCollisionMul",
                float(self.env_config.get('reward_collision_mul', 4)))
        env_sfp("rewardSplDeltaMul",
                float(self.env_config.get('reward_spl_delta_mul', 1)))
        env_sfp("segmentationMode",
                float(self.env_config.get('segmentation_mode', 1)))
        env_sfp("observationMode",
                float(self.env_config.get('obs_mode', 2)))
        env_sfp("episodeLength",
                float(self.env_config.get('episode_max_steps', 100)))
        env_sfp("selectedTaskIndex", float(self.env_config.get('task', 0)))
        env_sfp("goalSelectionIndex", float(self.env_config.get('goal', 0)))
        env_sfp("agentCarPhysics",
                float(self.env_config.get('agent_car_physics', 0)))
        env_sfp("goalDistance", float(self.env_config.get('goal_distance', 10)))
        env_sfp("numberOfTrafficVehicles",
                float(self.env_config.get('traffic_vehicles', 0)))

        env_path = self.env_config.get('env_path')
        env_path = None if env_path is None else str(Path(env_path).resolve())

        self._navsim_base_port = self.env_config.get('base_port', None)
        if self._navsim_base_port is None:
            self._navsim_base_port = UnityEnvironment.BASE_ENVIRONMENT_PORT if env_path else UnityEnvironment.DEFAULT_EDITOR_PORT
        self._navsim_worker_id = self.env_config.get('worker_id', 0)

        while True:
            try:
                log_folder = Path(
                    self.env_config.get('log_folder', './env_log')).resolve()
                log_folder.mkdir(parents=True, exist_ok=True)
                ad_args = [
                    "-force-device-index",
                    f"{self.env_config.get('env_gpu_id', 0)}",
                    "-observationWidth",
                    f"{self.env_config.get('obs_width', 256)}",
                    "-observationHeight",
                    f"{self.env_config.get('obs_height', 256)}",
                    "-fastForward", f"{self.start_from_episode - 1}",
                    "-showVisualObservations" if self.env_config.get(
                        'show_visual', False) else "",
                    "-saveStepLog" if self.debug else ""
                ]
                self.uenv = UnityEnvironment(file_name=env_path,
                                             log_folder=str(log_folder),
                                             seed=seed,
                                             timeout_wait=self.env_config.get(
                                                 'timeout', 600) + (0.5 * (
                                                     self.start_from_episode - 1)),
                                             worker_id=self._navsim_worker_id,
                                             base_port=self._navsim_base_port,
                                             no_graphics=False,
                                             side_channels=[eng_sc, env_pc,
                                                            self.map_side_channel,
                                                            self.fpc, self.nsc,
                                                            self.sapsc],
                                             additional_args=ad_args)
            except UnityWorkerInUseException:
                time.sleep(2)
                self._navsim_base_port += 1
            else:
                AroraGymEnv.logger.info(f"Created UnityEnvironment at port "
                                         f"{self._navsim_base_port + self._navsim_worker_id}"
                                         f" to start from episode {self.start_from_episode}")
                break

        super().__init__(unity_env=self.uenv,
                         uint8_visual=False,
                         flatten_branched=False,
                         allow_multiple_obs=True,
                         action_space_seed=seed
                         )

        # TODO: Once the environment has capability to start from an episode, then remove this
        # if self.start_from_episode > 1:
        #    logger.info(f'jumping to episode {self.start_from_episode}')
        # for i in range(1, self.start_from_episode):
        #    self.reset()
        #    logger.info(f'skipping episode {self.e_num}')

        # TODO: the filenames should be prefixed with specific id of this instance of env

        # TODO: Read the file upto start_episode and purge the records
        self.actions_file = log_folder / 'actions.csv'
        if self.save_actions:
            if (self.start_from_episode == 1) or (
                    self.actions_file.exists() == False):
                self.actions_file = self.actions_file.open(mode='w')
            else:
                self.actions_file = self.actions_file.open(mode='a')
            self.actions_writer = csv.writer(self.actions_file,
                                             delimiter=',',
                                             quotechar='"',
                                             quoting=csv.QUOTE_MINIMAL)

        if self.save_visual_obs and (self.obs_mode in [1, 2]):
            self.rgb_folder = log_folder / 'rgb_obs'
            self.rgb_folder.mkdir(parents=True, exist_ok=True)
            self.dep_folder = log_folder / 'dep_obs'
            self.dep_folder.mkdir(parents=True, exist_ok=True)
            self.seg_folder = log_folder / 'seg_obs'
            self.seg_folder.mkdir(parents=True, exist_ok=True)
        else:
            self.save_visual_obs = False

        if self.save_vector_obs and (self.obs_mode in [0, 2]):
            self.vec_file = log_folder / 'vec_obs.csv'
            if (self.start_from_episode == 1) or (
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
            self.save_vector_obs = False

    def _save_obs(self, obs):
        """Private method to save the observations in file

        Args:
            obs: observations object as returned by step() or reset()

        Returns:

        """
        if self.save_vector_obs:
            self.vec_writer.writerow(
                [self.e_num, self.s_num, self.spl_current, time.time()] +
                list(obs[-1]))
            self.vec_file.flush()
        if self.save_visual_obs:
            filename = f'{self.e_num}_{self.s_num}.jpg'
            imwrite(str(self.rgb_folder / filename), obs[0] * 255)
            imwrite(str(self.dep_folder / filename), obs[1] * 255)
            imwrite(str(self.seg_folder / filename), obs[2] * 255)

    def reset(self) -> Union[List[np.ndarray], np.ndarray]:
        s0 = super().reset()
        self.obs = s0
        if self.obs_mode in [0, 2]:
            vec_obs = list(self.obs[-1])
            self._agent_position = vec_obs[0:3]
            self._agent_velocity = vec_obs[3:6]
            self._agent_rotation = vec_obs[6:10]
            self._goal_position = vec_obs[10:13]
        self.s_num = 0
        self.e_num += 1 if self.e_num else self.start_from_episode
        self.spl_start = self.spl_current = self.shortest_path_length
        self._save_obs(self.obs)
        return s0

    def step(self, action: List[Any]) -> GymStepResult:
        s_, r, episode_done, info = super().step(action)
        self.obs = s_
        if self.obs_mode in [0, 2]:
            vec_obs = list(self.obs[-1])
            self._agent_position = vec_obs[0:3]
            self._agent_velocity = vec_obs[3:6]
            self._agent_rotation = vec_obs[6:10]
            self._goal_position = vec_obs[10:13]
        self.s_num += 1
        self.spl_current = self.shortest_path_length
        self._save_obs(self.obs)
        if self.save_actions:
            self.actions_writer.writerow(
                [self.e_num, self.s_num] + list(action))
            self.actions_file.flush()
        return s_, r, episode_done, info

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

    def render(self, mode='rgb_array'):
        """Returns the image array based on the render mode

        Args:
            mode: 'rgb_array' or 'depth' or 'segmentation' or 'vector'

        Returns:
            For Observation Mode 1 and 2: each render mode returns a numpy array of the image
            For Observation Mode 0 and 2: render mode vector returns vector observations
        """
        if mode == 'rgb_array' and self.obs_mode in [1, 2]:
            obs = self.obs[0]
        elif mode == 'depth' and self.obs_mode in [1, 2]:
            obs = self.obs[1]
        elif mode == 'segmentation' and self.obs_mode in [1, 2]:
            obs = self.obs[2]
        elif mode == 'vector' and self.obs_mode in [0, 2]:
            obs = self.obs[-1]
        else:
            raise ValueError("Bad render mode was specified or the "
                             "observation mode of the environment doesnt "
                             "support this render mode. render mode = "
                             "{mode}, observation mode = {self.obs_mode}")
        return obs

    def set_agent_state(self, position: Optional[List[float]] = None,
                        rotation: Optional[List[float]] = None):
        """Set the agent position or rotation

        Args:
            position: a list of x,y,z in Unity's coordinate system
            rotation: a list of x,y,z,w in Unity's coordinate system

        If the position or rotation is not provided as argument,
        then it takes them from the current state.

        Returns: True if the state is set, else False

        """

        #print("Agent Position", position)
        current_state = self.agent_state

        agent_id = 0
        #current_pos = self.agent_position if position is None else position
        #current_rot = self.agent_rotation if rotation is None else rotation
        #
        #state = np.concatenate(([agent_id], current_pos, current_rot))
        state = [agent_id]
        state += self.agent_position if position is None else position
        state += self.agent_rotation if rotation is None else rotation

        #print("State", state)

        self.uenv._process_immediate_message(
            self.sapsc.build_immediate_request("agentPosition",
                                               state))

        if self.sapsc.success:
            if position is not None:
                self._agent_position = position
            if rotation is not None:
                self._agent_rotation = rotation
        return self.sapsc.success

    def set_agent_position(self, position: Optional[List[float]]):
        """Set the agent position

        Args:
            position: a list of x,y,z in Unity's coordinate system

        If the position is not provided as argument,
        then it takes them from the current state.

        Returns: True if the state is set, else False

        """

        return self.set_agent_state(position=position)

    def set_agent_rotation(self, rotation: Optional[List[float]]):
        """Set the agent  rotation

        Args:
            rotation: a list of x,y,z,w in Unity's coordinate system

        If the rotation is not provided as argument,
        then it takes them from the current state.

        Returns: True if the state is set, else False

        """

        return self.set_agent_state(rotation=rotation)

    def get_navigable_map(self, resolution_x=256, resolution_y=256,
                          cell_occupancy_threshold=0.5) -> np.ndarray:
        """Get the Navigable Areas map

        Args:
            resolution_x: The size of the agent_x axis of the resulting grid, 1 to 3276, default = 256
            resolution_y: The size of the y axis of the resulting grid, 1 to 2662, default = 256
            cell_occupancy_threshold: If at least this much % of the cell is occupied, then it will be marked as non-navigable, 0 to 1.0, default = 50%

        Returns:
            A numpy array having 0 for non-navigable and 1 for navigable cells.

        Note:
            Largest resolution is 3284 x 2666
        """

        # TODO : Clean up these notes
        # The raw map array received from the Unity game is a row-major 1D flattened
        # bitpacked array with the y-axis data ordered for image output
        # (origin at top left).

        # For example, if reshaping to a 2D array without reordering with
        # dimensions `(resolution_y, resolution_x)`, then the desired coordinate `(x,y)`
        # is at array element `[resolution_y-1-y, x]`.
        # Finding the agent map position based on world position*:
        # `map_x = floor(world_x / (max_x / resolution_x) )`
        # `map_y = (resolution_y - 1) - floor(world_z / (max_y / resolution_y) )`

        # *Note: When converting from the 3-dimensional world position to the
        # 2-dimensional map, the world y-axis is omitted. The map's y-axis represents
        # the world's z-axis.

        if (resolution_x > 3284) or (resolution_y > 2666):
            raise ValueError("maximum map size is 3284 agent_x 2666")

        self.uenv._process_immediate_message(
            self.map_side_channel.build_immediate_request("binaryMap",
                                                          [resolution_x,
                                                           resolution_y,
                                                           cell_occupancy_threshold]))
        return self.map_side_channel.requested_map

        # def start_navigable_map(self, resolution_x=256, resolution_y=256,
        #                        cell_occupancy_threshold=0.5):
        # """Start the Navigable Areas map
        #
        # Args:
        #    resolution_x: The size of the agent_x axis of the resulting grid, default = 256
        #    resolution_y: The size of the y axis of the resulting grid, default = 256
        #    cell_occupancy_threshold: If at least this much % of the cell is occupied, then it will be marked as non-navigable, default = 50%
        #
        # Returns:
        #    Nothing

        # Note:
        #    Largest resolution is 3284 agent_x 2666
        # """
        #    if (resolution_x > 3284) or (resolution_y > 2666):
        #        raise ValueError("maximum map size is 3284 agent_x 2666")
        #    self.map_side_channel.send_request("binaryMap",
        #                                       [resolution_x, resolution_y,
        #                                        cell_occupancy_threshold])
        # print('Inside get navigable map function:',self.map_side_channel.requested_map)

        # def get_navigable_map(self) -> np.ndarray:
        # """Get the Navigable Areas map
        #
        # Args:
        #
        # Returns:
        #    A numpy array having 0 for non-navigable and 1 for navigable cells
        #
        # Note:
        #    This only works if you have called ``reset()`` or ``step()`` on the
        #    environment at least once after calling start_navigable_map() method.
        # """

    #    return self.map_side_channel.requested_map

    def unity_to_navmap_location(self, unity_x, unity_z, navmap_max_x=256,
                                 navmap_max_y=256):
        """Convert a location from Unity's 3D coordinate system to navigable map's 2D coordinate system

        Args:
            unity_x: x coordinate in unity
            unity_z: z coordinate in unity
            navmap_max_x: maximum x of navmap
            navmap_max_y: maximum y of navmap

        Returns:
            navmap_x, navmap_y
        """
        unity_max_x, _, unity_max_z = self.unity_map_dims
        # TODO: 0 <= unity_x < math.floor(unity_max_x) && 0 <= unity_z < math.floor(unity_max_z)
        navmap_x = math.floor(
            unity_x / (math.floor(unity_max_x) / navmap_max_x))
        navmap_y = math.floor(
            unity_z / (math.floor(unity_max_z) / navmap_max_y))
        return navmap_x, navmap_y

    def navmap_to_unity_location(self, navmap_x, navmap_y, navmap_max_x=256,
                                 navmap_max_y=256, navmap_cell_center=True):
        """Convert a location from navigable map's 2D coordinate system to Unity's 3D coordinate system

        Args:
            navmap_x, navmap_y: x, y location on navmap
            navmap_max_x, navmap_max_y: maximum x,y on navmap
            navmap_cell_center: Whether to return the point in cell center, default True.

        Returns:
            unity_x, unity_z
        """
        unity_max_x, unity_max_y, unity_max_z = self.unity_map_dims

        # TODO:  input: 0 <= navmap_x < navmap_max_x && 0<= navmap_y < navmap_max_y
        unity_x = navmap_x * (math.floor(unity_max_x) / navmap_max_x)
        unity_z = navmap_y * (math.floor(unity_max_z) / navmap_max_y)
        if navmap_cell_center:
            unity_x += (math.floor(unity_max_x) / navmap_max_x) / 2
            unity_z += (math.floor(unity_max_z) / navmap_max_y) / 2

        return unity_x, unity_z

    def unity_to_navmap_rotation(self, unity_rotation: List[float]):
        """Convert a rotation from Unity's quarternion to navigable map's 2D coordinate system

        Args:
            unity_rotation: [x,y,z,w] in unity quaternion system

        Returns:
            [x,y] vector components of rotation
        """
        x, _, z = self._qv_mult(unity_rotation, [0, 0, 1])
        return self._normalize([x, z])

    def unity_rotation_in_euler(self, unity_rotation: List[float] = None):
        """Position of agent in Euler coordinates roll_x, pitch_y, yaw_z

        Args:
            unity_rotation: [x,y,z,w] in unity quaternion system

        Returns:
            pitch_y, yaw_z, roll_x
        """
        if unity_rotation is None:
            unity_rotation = self.agent_rotation

        y_, z_, x_, w = unity_rotation
        # Convert a quaternion into euler angles (roll, pitch, yaw)
        # roll is rotation around x_ in radians (counterclockwise)
        # pitch is rotation around y_ in radians (counterclockwise)
        # yaw is rotation around z_ in radians (counterclockwise)

        t0 = +2.0 * (w * x_ + y_ * z_)
        t1 = +1.0 - 2.0 * (x_ * x_ + y_ * y_)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y_ - z_ * x_)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z_ + x_ * y_)
        t4 = +1.0 - 2.0 * (y_ * y_ + z_ * z_)
        yaw_z = math.atan2(t3, t4)

        return pitch_y, yaw_z, roll_x  # in radians
        
    def navmap_to_unity_rotation(self, navmap_rotation: List[float]):
        """Convert a rotation from navigable map's 2D coordinate system to Unity's quarternion

        Args:
            navmap_rotation: x,y vector components of rotation

        Returns:
            [x,y,z,w] Unity's quarternion
        """
        x, y = navmap_rotation

        v1 = self._normalize([x, 0, y])
        v2 = self._normalize(np.cross([0.1, 0], v1))
        v3 = np.cross(v1, v2)
        m00, m01, m02 = v2
        m10, m11, m12 = v3
        m20, m21, m22 = v1
        num8 = (m00 + m11) + m22
        if num8 > 0:
            num = math.sqrt(num8 + 1)
            w = num * 0.5
            num = 0.5 / num
            x = (m12 - m21) * num
            y = (m20 - m02) * num
            z = (m01 - m10) * num
        elif m00 >= m11 and m00 >= m22:
            num7 = math.sqrt(1 + m00 - m11 - m22)
            num4 = 0.5 / num7
            x = 0.5 / num7
            y = (m01 + m10) * num4
            z = (m02 + m20) * num4
            w = (m12 - m21) * num4
        elif m11 > m22:
            num6 = math.sqrt(1 + m11 - m00 - m22)
            num3 = 0.5 / num6
            x = (m10 + m01) * num3
            y = 0.5 * num6
            z = (m21 + m12) * num3
            w = (m20 - m02) * num3

        else:
            num5 = math.sqrt(1 + m22 - m00 - m11)
            num2 = 0.5 / num5
            x = (m20 + m02) * num2
            y = (m21 + m12) * num2
            z = 0.5 * num5
            w = (m01 - m10) * num2
        return [x, y, z, w]

    def _normalize(self, vec: List[float]):
        magnitude = 0.0
        for i in vec:
            magnitude += (i * i)
        magnitude = math.sqrt(magnitude)
        return vec / magnitude

    def _qv_mult(self, q: List[float], v: List[float]):
        qx, qy, qz, qw = q
        vx, vy, vz = v
        qc = [-q.x, -qy, -qz, qw]
        d = [vx, vy, vz, 0]
        result = self._q_mult(self._q_mult(q, d), qc)
        return result[0:3]

    def _q_mult(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q1

        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        return [x, y, z, w]

    def sample_navigable_point(self, x: float = None, y: float = None,
                               z: float = None):
        """Provides a random sample of navigable point

        Args:
            x,y,z: x,y,z in unity's coordinate system

        Returns:
            [] x,y,z all are None: returns a navigable point
            [x,z] only y is none: returns if x,z is navigable at some ht y
            [x,y,z] None of them are none: returns if x,y,z is navigable or not
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

        self.uenv._process_immediate_message(
            self.nsc.build_immediate_request("navigable", point))

        return self.nsc.point

    def is_navigable(self, x: float, y: float, z: float) -> bool:
        """Returns if the point is navigable or not

        Args:
            x,y,z: the unity coordinates of the point to check

        Returns:
            if the point represented by x,y,z is navigable or not
        """
        return self.sample_navigable_point(x, y, z)

    @staticmethod
    def register_with_gym():
        """Registers the environment with gym registry with the name navsim

        """
        env_id = 'arora-v0'
        from gym.envs.registration import register, registry

        env_dict = registry.env_specs.copy()
        for env in env_dict:
            if env_id in env:
                print(f"navsim_envs: Removing {env} from Gym registry")
                del registry.env_specs[env]

        print(f"navsim_envs: Adding {env_id} to Gym registry")
        register(id=env_id, entry_point='navsim_envs.env:AroraGymEnv')

    @staticmethod
    def register_with_ray():
        """Registers the environment with ray registry with the name navsim

        """
        from ray.tune.registry import register_env
        register_env("arora-v0", navsimgymenv_creator)

    def get_dummy_obs(self):
        """returns dummy observations

        Returns:

        """
        # prepare input data
        obs_names = []
        obs_data = []
        for state_dim in [obs.shape for obs in self.observation_space.spaces]:
            # obs_dim = [1] + [state_dim[2],state_dim[1],state_dim[0],
            obs = np.zeros([1] + list(state_dim))
            obs_name = 'state'
            for i in range(len(state_dim)):
                #    input_dim.append(state_dim[i])
                obs_name += f'_{state_dim[i]}'
            obs_names.append(obs_name)
            obs_data.append(obs)

        # print([o.shape for o in obs_data],obs_names)

        return obs_data, obs_names

    def get_dummy_actions(self):
        """returns dummy actions

        Returns:

        """
        action_dim = self.action_space.shape[0]
        actions_data = np.zeros([1, action_dim], dtype=np.float)
        actions_names = f'action_{action_dim}'
        return actions_data, actions_names

    # Functions added to have parity with Env and RLEnv of habitat lab
    @property
    def sim(self):
        """Returns an instance of the sim

        Added for compatibility with habitat API.

        Returns: link to self

        """
        return self

    @property
    def unity_map_dims(self):
        """Returns the maximum x,y,z values of Unity Map

        Note: While converting to 2-D map, the Z-axis max of 3-D Unity Map
        corresponds to Y-axis max of 2-D map

        Returns: maximum x, y, z from unity map

        """
        return 3284.0, 52.9, 2666.3

    @property
    def agent_state(self):
        """Agent state is position (x,y,z), rotation (x,y,,z,w)
        """
        return self.agent_position, self.agent_rotation

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

    @property
    def shortest_path_length(self):
        """the shortest navigable path length from current location to
        goal position
        """
        return self.fpc.get_property("ShortestPath")

### Position Scan - Not Available
# Given a position and this returns the attribution data of the first object
# found at the given position. Objects are searched for within a 1 meter radius
# of the given position. If the position is not loaded in the environment then
# None will be returned.
