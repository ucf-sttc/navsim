import csv
from pathlib import Path
import numpy as np
import uuid

from typing import Any, Dict, List, Optional, Tuple, Union

import time

from gym_unity.envs import (
    UnityToGymWrapper,
    GymStepResult
)

from mlagents_envs.logging_util import get_logger

from mlagents_envs.exception import UnityWorkerInUseException
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

logger = get_logger(__name__)

try:
    from cv2 import imwrite as imwrite

    logger.info("Navsim is using cv2 as image library")
except:
    try:
        from imageio import imwrite as imwrite

        logger.info("Navsim is using imageio as image library")
    except:
        try:
            from matplotlib.pyplot import imsave as imwrite

            logger.info("Navsim is using matplotlib as image library")
        except:
            try:
                from PIL import Image

                logger.info("Navsim is using PIL as image library")


                def imwrite(filename, arr):
                    im = Image.fromarray(arr)
                    im.save(filename)
            except:
                def imwrite(filename=None, arr=None):
                    logger.warning("Navsim is unable to load any of the following "
                                  "image libraries: cv2, imageio, matplotlib, "
                                  "PIL. Install one of these libraries to "
                                  "save visuals.")

                imwrite()


def navsimgymenv_creator(env_config):
    return NavSimGymEnv(env_config)  # return an env instance


class NavSimGymEnv(UnityToGymWrapper):
    """NavSimGymEnv Class is a wrapper to Unity2Gym that inherits from the Gym interface

    Read the **NavSim Environment Tutorial** on how to use this class.
    """
    metadata = {'render.modes': ['rgb_array', 'depth', 'segmentation']}

    def __init__(self, env_config) -> None:
        """
        env_config: The environment configuration dictionary or ObjDict Object
        """

        # TODO: convert env_config to self.env_config so we can add missing values
        #   and use self.env_config to print in the info section
        # filename: Optional[str] = None, obs_mode: int = 0, max_steps:int = 5):
        self.env_config = env_config
        self.obs_mode = int(self.env_config.get('obs_mode', 2))
        self.start_from_episode = int(
            self.env_config.get('start_from_episode', 1))
        self.debug = env_config.get("debug", False)
        self.obs = None

        if self.obs_mode == 0:
            env_config["save_visual_obs"] = False
        elif self.obs_mode == 1:
            env_config["save_vector_obs"] = False

        self.save_visual_obs = env_config.get("save_visual_obs", False)
        self.save_vector_obs = env_config.get("save_vector_obs", False)

        self.e_num = 0
        self.s_num = 0

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

        # print(self.map_side_channel)

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
        env_sfp("goalDistance", float(self.env_config.get('goal_distance', 50)))

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
                    f"-force-device-index {self.env_config.get('env_gpu_id', 0)}",
                    f"-observationWidth {self.env_config.get('obs_width', 256)}",
                    f"-observationHeight {self.env_config.get('obs_height', 256)}"
                ]
                uenv = UnityEnvironment(file_name=env_path,
                                        log_folder=str(log_folder),
                                        seed=seed,
                                        timeout_wait=self.env_config.get(
                                            'timeout', 600),
                                        worker_id=self._navsim_worker_id,
                                        base_port=self._navsim_base_port,
                                        no_graphics=False,
                                        side_channels=[eng_sc, env_pc,
                                                       self.map_side_channel,
                                                       self.fpc],
                                        additional_args=ad_args)
            except UnityWorkerInUseException:
                time.sleep(2)
                self._navsim_base_port += 1
            else:
                logger.info(f"Created UnityEnvironment at port "
                            f"{self._navsim_base_port + self._navsim_worker_id}")
                break

        super().__init__(unity_env=uenv,
                         uint8_visual=False,
                         flatten_branched=False,
                         allow_multiple_obs=True,
                         action_space_seed=seed
                         )

        # TODO: Once the environment has capability to start from an episode, then remove this
        if self.start_from_episode > 1:
            logger.info(f'jumping to episode {self.start_from_episode}')
        for i in range(1, self.start_from_episode):
            self.reset()
            logger.info(f'skipping episode {self.e_num}')

        # TODO: the filenames should be prefixed with specific id of this instance of env

        if self.save_visual_obs or self.save_vector_obs:
            # TODO: Read the file upto start_episode and purge the records
            self.actions_file = log_folder / 'actions.csv'
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

    def save_obs(self, obs):
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
        result = super().reset()

        self.e_num += 1
        self.s_num = 0
        if self.e_num >= self.start_from_episode:
            self.obs = result
            self.spl_start = self.spl_current = self.get_shortest_path_length()
            self.save_obs(self.obs)
        return result

    def step(self, action: List[Any]) -> GymStepResult:
        s_, r, episode_done, info = super().step(action)
        self.obs = s_
        self.s_num += 1
        if self.save_vector_obs or self.save_visual_obs:
            self.actions_writer.writerow(
                [self.e_num, self.s_num] + list(action))
            self.actions_file.flush()
        self.spl_current = self.get_shortest_path_length()
        self.save_obs(self.obs)
        return s_, r, episode_done, info

    #def close(self):
    #    if self.save_vector_obs:
    #        self.vec_file.close()
    #    if self.save_vector_obs or self.save_visual_obs:
    #        self.actions_file.close()
    #    super().close()

    def info(self):
        """Prints the information about the environment

        """
        print('-----------')
        print("Env Info")
        print('-----------')
        if self.spec is not None:
            print(self.spec.id)
        print('Action Space:', self.action_space)
        print('Action Space Shape:', self.action_space.shape)
        print('Action Space Low:', self.action_space.low)
        print('Action Space High:', self.action_space.high)
        print('Observation Mode:', self.obs_mode)
        # print('Gym Observation Space:', self.genv.observation_space)
        # print('Gym Observation Space Shape:', self.genv.observation_space.shape)
        print('Observation Space:', self.observation_space)
        print('Observation Space Shape:', self.observation_space.shape)
        print('Observation Space Shapes:', self.observation_space_shapes)
        print('Observation Space Types:', self.observation_space_types)
        print('Reward Range:', self.reward_range)
        print('Metadata:', self.metadata)
        print('--------------------------------------')
        return self

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
    def observation_space_shapes(self) -> list:
        """Returns the dimensions of the observation space
        """
        return [obs.shape for obs in self.observation_space.spaces]

    @property
    def observation_space_types(self) -> list:
        """Returns the dimensions of the observation space
        """
        return [type(obs) for obs in self.observation_space.spaces]

    def render(self, mode='rgb_array') -> None:
        """Returns the image array based on the render mode

        Args:
            mode: 'rgb_array' or 'depth' or 'segmentation'

        Returns:
            For Observation Modes 1 and 2:
                For each render mode returns a numpy array of the image.
            For Observation Mode 0:
                None
        """
        if self.obs_mode in [1, 2]:
            if mode == 'rgb_array':
                obs = self.obs[0]
            elif mode == 'depth':
                obs = self.obs[1]
            elif mode == 'segmentation':
                obs = self.obs[2]
        else:
            obs = self.obs[-1]
        return obs

    def start_navigable_map(self, resolution_x=256, resolution_y=256,
                            cell_occupancy_threshold=0.5):
        """Start the Navigable Areas map

        Args:
            resolution_x: The size of the x axis of the resulting grid, default = 256
            resolution_y: The size of the y axis of the resulting grid, default = 256
            cell_occupancy_threshold: If at least this much % of the cell is occupied, then it will be marked as non-navigable, default = 50%

        Returns:


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

        Returns:
            A numpy array having 0 for non-navigable and 1 for navigable cells

        Note:
            This only works if you have called ``reset()`` or ``step()`` on the
            environment at least once after calling start_navigable_map() method.
        """
        return self.map_side_channel.requested_map

    def get_shortest_path_length(self):
        return self.fpc.get_property("ShortestPath")

    def get_navigable_map_point_from_unity_map_point(self, x, y):
        map_x = FloorToInt(x / (unity_max_x / map_max_x))
        map_y = FloorToInt(y / (unity_max_y / map_max_y))

    @staticmethod
    def register_with_gym():
        """Registers the environment with gym registry with the name navsim

        """
        id = 'navsim-v0'
        from gym.envs.registration import register, registry

        env_dict = registry.env_specs.copy()
        for env in env_dict:
            if id in env:
                logger.info(f"Removing {env} from Gym registry")
                del registry.env_specs[env]

        logger.info(f"Adding {id} to Gym registry")
        register(id=id, entry_point='navsim:NavSimGymEnv')

    @staticmethod
    def register_with_ray():
        """Registers the environment with ray registry with the name navsim

        """
        from ray.tune.registry import register_env
        register_env("navsim-v0", navsimgymenv_creator)

    def get_dummy_obs(self):
        # prepare input data
        obs_names = []
        obs_data = []
        for state_dim in self.observation_space_shapes:
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
        action_dim = self.action_space.shape[0]
        actions_data = np.zeros([1, action_dim], dtype=np.float)
        actions_names = f'action_{action_dim}'
        return actions_data, actions_names

        """
        # prepare input data
        input_data = []
        input_names = []
        for state_dim in self.env.observation_space_shapes:
            if len(state_dim) == 1:
                random_input = torch.randn(1, state_dim[0]).to(device)
                input_name = f'state_{state_dim[0]}'
            else:  # visual
                random_input = torch.randn(1, state_dim[2], state_dim[0],
                                           state_dim[1]).to(device)
                input_name = f'state_{state_dim[0]}_{state_dim[1]}_{state_dim[2]}'

            input_data.append(random_input)
            input_names.append(input_name)

        # export actor
        model = self.actor
        torch.onnx.export(model,
                          args=input_data,
                          f=f"{folder}/actor.onnx",
                          export_params=True,
                          opset_version=9,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=['action'])

        if critic:
            # add action data for critic
            action_dim = self.env.action_space_shape[0]
            random_input = torch.randn(1, action_dim).to(device)
            input_name = f'action_{action_dim}'
            input_data = [input_data]

            # print(len(input_data))
            input_data.append(random_input)
            input_data = tuple(input_data)
            # print(len(input_data))
            input_names.append(input_name)

            # export critic
            model = self.critic
            torch.onnx.export(model,
                              args=input_data,
                              f=f"{folder}/critic.onnx",
                              export_params=True,
                              opset_version=9,
                              do_constant_folding=True,
                              input_names=input_names,
                              output_names=['q'])
        """

    # Functions added to have parity with Env and RLEnv of habitat lab
    @property
    def sim(self):
        """Returns itself

        Added for compatibility with habitat API.

        Returns: link to self

        """
        return self

    @property
    def get_unity_map_dims(self):
        return 3276.8, 2662.4


# sim.get_agent_state() -> x, y, orientation
# sim.set_agent_state(position, orientation)
# sim.get_observations_at(position, orientation) -> observation when agent is at position with specified orientation
# sim.sample_navigable_point() -> x,y (must be a navigable location in the map)

class MapSideChannel(SideChannel):
    """This is the SideChannel for retrieving map data from Unity.
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
        """Sends a request to Unity
        The arguments for a mapRequest are ("binaryMap", [RESOLUTION_X, RESOLUTION_Y, THRESHOLD])
        """
        self.resolution = value
        msg = OutgoingMessage()
        msg.write_string(key)
        msg.write_float32_list(value)
        super().queue_message_to_send(msg)
