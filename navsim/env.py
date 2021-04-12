from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from pathlib import Path
from typing import List, Any
import numpy as np
import uuid
import navsim
import cv2

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

        env_conf = ObjDict({
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
    def __init__(self, conf: navsim.util.ObjDict) -> None:
        """
        conf: ObjDict having Environment Conf
        :param conf:
        """
        # filename: Optional[str] = None, observation_mode: int = 0, max_steps:int = 5):
        self.conf = conf
        self.observation_mode = self.conf['observation_mode']

        self.uenv = None
        self.uenv = self.__open_uenv()
        super().__init__(self.uenv, False, False, True)
                    # (Env, uint8_visual, flatten_branched, allow_multiple_obs)
        #self.seed(self.conf['seed']) # unity-gym env seed is not working, seed has to be passed with unity env

    def __open_uenv(self) -> UnityEnvironment:
        if self.uenv:
            raise ValueError('Environment already open')
        else:
            log_folder = Path(self.conf['log_folder'])
            log_folder.mkdir(parents=True, exist_ok=True)

            engine_side_channel = EngineConfigurationChannel()
            environment_side_channel = EnvironmentParametersChannel()
            self.map_side_channel = MapSideChannel()
            #print(self.map_side_channel)
            engine_side_channel.set_configuration_parameters(time_scale=10, quality_level=0)

            environment_side_channel.set_float_parameter("rewardForGoalCollision", self.conf['reward_for_goal'])
            environment_side_channel.set_float_parameter("rewardForExplorationPointCollision",
                                                         self.conf['reward_for_ep'])
            environment_side_channel.set_float_parameter("rewardForOtherCollision", self.conf['reward_for_other'])
            environment_side_channel.set_float_parameter("rewardForFallingOffMap",
                                                         self.conf['reward_for_falling_off_map'])
            environment_side_channel.set_float_parameter("rewardForEachStep", self.conf['reward_for_step'])
            environment_side_channel.set_float_parameter("segmentationMode", self.conf['segmentation_mode'])
            environment_side_channel.set_float_parameter("observationMode", self.conf['observation_mode'])
            environment_side_channel.set_float_parameter("episodeLength", self.conf['max_steps'])
            environment_side_channel.set_float_parameter("selectedTaskIndex", self.conf['task'])
            environment_side_channel.set_float_parameter("goalSelectionIndex", self.conf['goal'])
            environment_side_channel.set_float_parameter("agentCarPhysics", self.conf['agent_car_physics'])
            environment_side_channel.set_float_parameter("goalDistance", self.conf['goal_distance'])

            uenv_file_name = str(Path(self.conf['env_path']).resolve()) if self.conf['env_path'] else None
            self.uenv = UnityEnvironment(file_name=uenv_file_name,
                                         log_folder=str(log_folder.resolve()),
                                         seed=self.conf['seed'],
                                         timeout_wait=self.conf['timeout'],
                                         worker_id=self.conf['worker_id'],
                                         # base_port=self.conf['base_port'],
                                         no_graphics=False,
                                         side_channels=[engine_side_channel, environment_side_channel,self.map_side_channel])

            return self.uenv

    """
    def close_uenv(self):
        if self.uenv is None:
            print('uenv is None')
        else:
            self.uenv.close()
    """

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
        #print('Gym Observation Space:', self.genv.observation_space)
        #print('Gym Observation Space Shape:', self.genv.observation_space.shape)
        print('Observation Space:', self.observation_space)
        print('Observation Space Shape:', self.observation_space.shape)
        print('Observation Space Shapes:', self.observation_space_shapes)
        print('Observation Space Types:', self.observation_space_types)
        print('Reward Range:', self.reward_range)
        print('Metadata:', self.metadata)
        return self

    def info_steps(self, save_visuals = False):
        """Prints the initial state, action sample, first step state

        """
        print('Initial State:', self.reset())
        action_sample = self.action_space.sample()
        print('Action sample:', action_sample)
        s,a,r,s_ = self.step(action_sample)
        print('First Step s,a,r,s_:', s,a,r,s_)
        if (self.observation_mode==1) or (self.observation_mode ==2):
            for i in range(0,3):
                cv2.imwrite(f'visual_{i}.jpg',(s[i]*255).astype('uint8'))
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

    def render(self,mode='') -> None:
        """Not implemented yet

        Args:
            mode:

        Returns:

        """
        pass

    def start_navigable_map(self,resolution_x=200,resolution_y=200,cell_occupancy_threshold=0.5):
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
        self.map_side_channel.send_request("binaryMap", [resolution_x,resolution_y,cell_occupancy_threshold])
        #print('Inside get navigable map function:',self.map_side_channel.requested_map)

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

    # Functions added to have parity with Env and RLEnv of habitat lab
    @property
    def sim(self):
        """Returns itself

        Added for compatibility with habitat API.

        Returns: link to self

        """
        return self

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)


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

        #img = Image.frombuffer('L', (self.resolution[0],self.resolution[1]), np.array(msg.get_raw_bytes())).convert('1')
        #timestr = time.strftime("%Y%m%d-%H%M%S")
        #img.save("img-"+timestr+".png")

        #raw_bytes = msg.get_raw_bytes()
        #unpacked_array = np.unpackbits(raw_bytes)[0:self.resolution[0]*self.resolution[1]]

        raw_bytes = msg.get_raw_bytes()
        self.requested_map  = np.unpackbits(raw_bytes)[0:self.resolution[0]*self.resolution[1]]
        self.requested_map = self.requested_map.reshape((self.resolution[0],self.resolution[1]))
        #self.requested_map = np.array(msg.get_raw_bytes()).reshape((self.resolution[0],self.resolution[1]))
        #print('map inside on message received:',self.requested_map, self.requested_map.shape)
        return self.requested_map

        #np.savetxt("arrayfile", np.asarray(img), fmt='%1d', delimiter='')

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
