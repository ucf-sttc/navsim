import csv
from pathlib import Path
from typing import List, Any
import numpy as np
import uuid
import cv2

from typing import Any, Dict, List, Optional, Tuple, Union

from gym_unity.envs import (
    UnityToGymWrapper,
    GymStepResult
)

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
from mlagents_envs.side_channel.engine_configuration_channel import \
    EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import \
    EnvironmentParametersChannel

from mlagents_envs.side_channel.float_properties_channel import \
    FloatPropertiesChannel


def navsimgymenv_creator(env_config):
    return NavSimGymEnv(env_config)  # return an env instance


class NavSimGymEnv(UnityToGymWrapper):
    """NavSimGymEnv Class is a wrapper to Unity2Gym that inherits from the Gym interface

    The configuration provided is as follows:

    Observation Mode

    * Vector 0

    * Visual 1

    * VectorVisual 2

    Segmentation Mode

    * Object Segmentation 0

    * Tag Segmentation 1

    * Layer Segmentation 2

    Task

    * PointNav 0

    * SimpleObjectNav 1

    * ObjectNav 2

    Goal

    * 0 - Tocus

    * 1 - sedan1

    * 2 - Car1

    * 3 - Car2

    * 4 - City Bus

    * 5 - Sporty_Hatchback

    * Else - SEDAN

    .. code-block:: python

        env_config = ObjDict({
            "log_folder": "unity.log",
            "seed": 123,
            "timeout": 600,
            "worker_id": 0,
            "base_port": 5005,
            "observation_mode": 2,
            "segmentation_mode": 1,
            "task": 0,
            "goal": 0,
            "goal_distance":50
            "max_steps": 10,
            "reward_for_goal": 50,
            "reward_for_ep": 0.005,
            "reward_for_other": -0.1,
            "reward_for_falling_off_map": -50,
            "reward_for_step": -0.0001,
            "agent_car_physics": 0,
            "episode_max_steps": 10,
            "start_from_episode":1,
            "env_path":args["env_path"]
        })

    Action Space: [Throttle, Steering, Brake]

    * Throttle: -1.0 to 1.0

    * Steering: -1.0 to 1.0

    * Brake: 0.0 to 1.0

    Observation Space: [[Raw Agent Camera],[Depth Agent Camera],[Segmentation Agent Camera],[Agent Position, Agent Velocity, Agent Rotation, Goal Position]]
    The vector observation space:
        Agent_Position.x, Agent_Position.y, Agent_Position.z,
        Agent_Velocity.x, Agent_Velocity.y, Agent_Velocity.z,
        Agent_Rotation.x, Agent_Rotation.y, Agent_Rotation.z, Agent_Rotation.w,
        Goal_Position.x, Goal_Position.y, Goal_Position.z
    """

    def __init__(self, env_config) -> None:
        """
        env_config: The environment configuration dictionary or ObjDict Object
        """
        # filename: Optional[str] = None, observation_mode: int = 0, max_steps:int = 5):
        self.env_config = env_config
        self.observation_mode = int(self.env_config.get('observation_mode', 2))
        self.start_from_episode = int(self.env_config.get('start_from_episode', 1))
        self.debug = env_config.get("debug", False)
        self.run_base_folder_str = env_config.get("run_base_folder_str", '.')
        self.run_base_folder = Path(self.run_base_folder_str)
        seed = int(self.env_config.get('seed'))

        # if self._env:
        #    raise ValueError('Environment already open')
        # else:
        log_folder = Path(self.env_config.get('log_folder', '.'))
        log_folder.mkdir(parents=True, exist_ok=True)

        self.map_side_channel = MapSideChannel()
        self.fpc = FloatPropertiesChannel()

        # print(self.map_side_channel)

        eng_sc = EngineConfigurationChannel()
        eng_sc.set_configuration_parameters(time_scale=10, quality_level=0)

        env_pc = EnvironmentParametersChannel()
        env_sfp = env_pc.set_float_parameter

        env_sfp("rewardForGoalCollision",
                float(self.env_config.get('reward_for_goal', 50)))
        env_sfp("rewardForExplorationPointCollision",
                float(self.env_config.get('reward_for_ep', 0.005)))
        env_sfp("rewardForOtherCollision",
                float(self.env_config.get('reward_for_other', -0.1)))
        env_sfp("rewardForFallingOffMap",
                float(self.env_config.get('reward_for_falling_off_map', -50)))
        env_sfp("rewardForEachStep",
                float(self.env_config.get('reward_for_step', -0.0001)))
        env_sfp("segmentationMode", float(self.env_config.get('segmentation_mode', 1)))
        env_sfp("observationMode", float(self.env_config.get('observation_mode', 2)))
        env_sfp("episodeLength", float(self.env_config.get('episode_max_steps', 100)))
        env_sfp("selectedTaskIndex", float(self.env_config.get('task', 0)))
        env_sfp("goalSelectionIndex", float(self.env_config.get('goal', 0)))
        env_sfp("agentCarPhysics", float(self.env_config.get('agent_car_physics', 0)))
        env_sfp("goalDistance", float(self.env_config.get('goal_distance', 50)))

        env_path = self.env_config.get('env_path')
        env_path = None if env_path is None else str(Path(env_path).resolve())

        uenv = UnityEnvironment(file_name=env_path,
                                log_folder=str(log_folder.resolve()),
                                seed=seed,
                                timeout_wait=self.env_config.get(
                                    'timeout', 600),
                                worker_id=self.env_config.get(
                                    'worker_id', 0),
                                # base_port=self.conf['base_port'],
                                no_graphics=False,
                                side_channels=[eng_sc, env_pc,
                                               self.map_side_channel, self.fpc])

        super().__init__(unity_env=uenv,
                         uint8_visual=False,
                         flatten_branched=False,
                         allow_multiple_obs=True,
                         action_space_seed=seed
                         )

        # TODO: Once the environment has capability to start from an episode, then remove this
        for i in range(1, self.start_from_episode):
            self.reset()
            
        # (Env, uint8_visual, flatten_branched, allow_multiple_obs)
        # self.seed(self.conf['seed']) # unity-gym env seed is not working, seed has to be passed with unity env
        if self.debug:
            # open the debug files
            # TODO: the filenames should be prefixed with specific id of this instance of env
            self.file_mode = 'w+'
            self.actions_file = open(
                str((self.run_base_folder / 'actions.txt').resolve()),
                mode=self.file_mode)
            self.observations_file = open(
                str((self.run_base_folder / 'observations.txt').resolve()),
                mode=self.file_mode)
            self.actions_writer = csv.writer(self.actions_file,
                                             delimiter=',',
                                             quotechar='"',
                                             quoting=csv.QUOTE_MINIMAL)
            self.observations_writer = csv.writer(self.observations_file,
                                                  delimiter=',',
                                                  quotechar='"',
                                                  quoting=csv.QUOTE_MINIMAL)

    def step(self, action: List[Any]) -> GymStepResult:
        result = super().step(action)

        if self.debug:
            self.actions_writer.writerow(action)
            self.actions_file.flush()
            if self.observation_mode in [0, 2]:
                vector_obs = result[0][-1]
                self.observations_writer.writerow(vector_obs)
                self.observations_file.flush()

        return result

    def info(self):
        """Prints the information about the environment

        """
        print('-----------')
        print("Env Info")
        print('-----------')
        if self.spec is not None:
            print(self.genv.spec.id)
        print('Action Space:', self.action_space)
        print('Action Space Shape:', self.action_space.shape)
        print('Action Space Low:', self.action_space.low)
        print('Action Space High:', self.action_space.high)
        print('Observation Mode:', self.observation_mode)
        # print('Gym Observation Space:', self.genv.observation_space)
        # print('Gym Observation Space Shape:', self.genv.observation_space.shape)
        print('Observation Space:', self.observation_space)
        print('Observation Space Shape:', self.observation_space.shape)
        print('Observation Space Shapes:', self.observation_space_shapes)
        print('Observation Space Types:', self.observation_space_types)
        print('Reward Range:', self.reward_range)
        print('Metadata:', self.metadata)
        return self

    def info_steps(self, save_visuals=False):
        """Prints the initial state, action sample, first step state

        """
        print('Initial State:', self.reset())
        action_sample = self.action_space.sample()
        print('Action sample:', action_sample)
        s, a, r, s_ = self.step(action_sample)
        print('First Step s,a,r,s_:', s, a, r, s_)
        if (self.observation_mode == 1) or (self.observation_mode == 2):
            for i in range(0, 3):
                cv2.imwrite(f'visual_{i}.jpg', (s[i] * 255).astype('uint8'))
        self.reset()
        return self

    @property
    def observation_space_shapes(self) -> list:
        """Returns the dimensions of the observation space
        """
        return [obs.shape for obs in self.observation_space.spaces]

    @property
    def observation_space_types(self) -> list:
        """Returns the dimensions of the observation space
        """
        return [type(obs) for obs in self.observation_space.spaces]

    def render(self, mode='') -> None:
        """Not implemented yet

        Args:
            mode:

        Returns:

        """
        pass

    def start_navigable_map(self, resolution_x=200, resolution_y=200,
                            cell_occupancy_threshold=0.5):
        """Get the Navigable Areas map

        Args:
            resolution_x: The size of the x axis of the resulting grid, default = 200
            resolution_y: The size of the y axis of the resulting grid, default = 200
            cell_occupancy_threshold: If at least this much % of the cell is occupied, then it will be marked as non-navigable, default = 50%

        Returns:
            A numpy array which has 0 for non-navigable and 1 for navigable cells

        Note:
            This method only works if you have called ``reset()`` or ``step()`` on the environment at least once.

        Note:
            Largest resolution that was found to be working was 2000 x 2000
        """
        self.map_side_channel.send_request("binaryMap",
                                           [resolution_x, resolution_y,
                                            cell_occupancy_threshold])
        # print('Inside get navigable map function:',self.map_side_channel.requested_map)

    def get_navigable_map(self) -> np.ndarray:
        """Get the Navigable Areas map

        Args:
            resolution_x: The size of the x axis of the resulting grid, default = 200
            resolution_y: The size of the y axis of the resulting grid, default = 200
            cell_occupancy_threshold: If at least this much % of the cell is occupied, then it will be marked as non-navigable, default = 50%

        Returns:
            A numpy array which has 0 for non-navigable and 1 for navigable cells

        Note:
            This method only works if you have called ``reset()`` or ``step()`` on the environment at least once.

        Note:
            Largest resolution that was found to be working was 2000 x 2000
        """
        return self.map_side_channel.requested_map

    def get_shortest_path_length(self):
        return self.fpc.get_property("ShortestPath")

    # Functions added to have parity with Env and RLEnv of habitat lab
    @property
    def sim(self):
        """Returns itself

        Added for compatibility with habitat API.

        Returns: link to self

        """
        return self

    @staticmethod
    def register_with_gym():
        """Registers the environment with gym registry with the name navsim

        """
        try:
            from gym.envs.registration import register
            register(id='navsim', entry_point='navsim.NavSimGymEnv')
        except Exception as e:
            print("Can not register NavSim Environment with Gym")
            print(e.message)

    @staticmethod
    def register_with_ray():
        """Registers the environment with ray registry with the name navsim

        """
        try:
            from ray.tune.registry import register_env
            register_env("navsim", navsimgymenv_creator)
        except Exception as e:
            print("Can not register NavSim Environment with Ray")
            print(e.message)


class MapSideChannel(SideChannel):
    """
    This is the SideChannel for retrieving map data from Unity.
    You can send map requests to Unity using send_request.
    The arguments for a mapRequest are ("binaryMap", [RESOLUTION_X, RESOLUTION_Y, THRESHOLD])
    """
    resolution = []

    def __init__(self) -> None:
        channel_id = uuid.UUID("24b099f1-b184-407c-af72-f3d439950bdb")
        super().__init__(channel_id)
        self.requested_map = None

    def on_message_received(self, msg: IncomingMessage) -> np.ndarray:
        if self.resolution is None:
            return None

        # mode as grayscale 'L' and convert to binary '1' because for some reason using only '1' doesn't work (possible bug)

        # img = Image.frombuffer('L', (self.resolution[0],self.resolution[1]), np.array(msg.get_raw_bytes())).convert('1')
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # img.save("img-"+timestr+".png")

        # raw_bytes = msg.get_raw_bytes()
        # unpacked_array = np.unpackbits(raw_bytes)[0:self.resolution[0]*self.resolution[1]]

        raw_bytes = msg.get_raw_bytes()
        self.requested_map = np.unpackbits(raw_bytes)[
                             0:self.resolution[0] * self.resolution[1]]
        self.requested_map = self.requested_map.reshape(
            (self.resolution[0], self.resolution[1]))
        # self.requested_map = np.array(msg.get_raw_bytes()).reshape((self.resolution[0],self.resolution[1]))
        # print('map inside on message received:',self.requested_map, self.requested_map.shape)
        return self.requested_map

        # np.savetxt("arrayfile", np.asarray(img), fmt='%1d', delimiter='')

    def send_request(self, key: str, value: List[float]) -> None:
        """
        Sends a request to Unity
        The arguments for a mapRequest are ("binaryMap", [RESOLUTION_X, RESOLUTION_Y, THRESHOLD])
        """
        self.resolution = value
        msg = OutgoingMessage()
        msg.write_string(key)
        msg.write_float32_list(value)
        super().queue_message_to_send(msg)
