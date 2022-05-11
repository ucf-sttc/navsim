#RIDE

import numpy as np

from typing import Any, List, Union, Optional

import math

from .configs import default_env_config

# @attr.s(auto_attribs=True)
# class AgentState:
#    position: Optional["np.ndarray"]
#    rotation: Optional["np.ndarray"] = None

from navsim_envs.exceptions import EnvNotInitializedError
from .unity_env import RideUnityEnv

from mlagents_envs.rpc_utils import steps_from_proto

from navsim_envs.base_envs import AroraGymEnvBase

def ridegymenv_creator(env_config):
    return RideGymEnv(env_config)  # return an env instance


def _normalize(vec: np.ndarray):
    # magnitude = 0.0
    # for i in vec:
    #    magnitude += (i * i)
    # magnitude =   # math.sqrt(magnitude)
    return vec / np.linalg.norm(vec)


def _q_mult(q1: List[float], q2: List[float]):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return [x, y, z, w]


def _qv_mult(q: List[float], v: List[float]):
    qx, qy, qz, qw = q
    vx, vy, vz = v
    qc = [-qx, -qy, -qz, qw]
    d = [vx, vy, vz, 0]
    result = _q_mult(_q_mult(q, d), qc)
    return result[0:3]


class RideGymEnv(AroraGymEnvBase):
    """RideGymEnv inherits from Unity2Gym that inherits from the Gym interface.

    Read the **NavSim Environment Tutorial** on how to use this class.
    """
    metadata = {
        'render.modes': ['rgb_array','vector']}

    def __init__(self, env_config) -> None:
        """
        env_config: The environment configuration dictionary Object
        """

        super().__init__(env_config, default_env_config, RideUnityEnv)

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

    def render(self, mode:str='rgb_array'):
        """Returns the image array based on the render mode

        Args:
            mode: 'rgb_array' or 'depth' or 'segmentation' or 'vector'

        Returns:
            For Observation Mode 1 and 2 - each render mode returns a numpy array of the image
            For Observation Mode 0 and 2 - render mode vector returns vector observations
        """
        if mode == 'rgb_array' and self.env_config["obs_mode"] in [1]:
            obs = self._obs[0]
        elif mode == 'depth' and self.env_config["obs_mode"] in [1]:
            obs = self._obs[1]
        elif mode == 'segmentation' and self.env_config["obs_mode"] in [1]:
            obs = self._obs[2]
        elif mode == 'vector' and self.env_config["obs_mode"] in [0, 1]:
            obs = self._obs[-1]
        else:
            raise ValueError(f"Bad render mode was specified or the "
                             f"observation mode of the environment doesnt "
                             f"support this render mode. render mode = "
                             f"{mode}, observation mode = {self.env_config['obs_mode']}")
        return obs

    def get_agent_obs_at(self, position: Optional[List[float]] = None,
                         rotation: Optional[List[float]] = None):
        """Get the agent observations at position or rotation

        Args:
            position: a list of x,y,z in Unity's coordinate system
            rotation: a list of x,y,z,w in Unity's coordinate system

        If the position or rotation is not provided as argument,
        then it takes them from the current state.

        Returns:
            Observations if possible, else None

        """

        # Just added to cause exception
        pos, rot = self.agent_position, self.agent_rotation

        agent_id = 0
        state = [agent_id]
        state += pos if position is None else position
        state += rot if rotation is None else rotation

        unity_output = self.uenv.process_immediate_message(
            self.uenv.sapsc.build_immediate_request("getObservation",
                                                    state))

        if self.uenv.sapsc.success:
            s_, _, _, _ = self._single_step(
                steps_from_proto(
                    unity_output.rl_output.agentInfos["immediate"].value,
                    self.uenv._env_specs[self.name]
                )[1]
            )
            # self.obs = s_
            # self._set_obs()
            # if position is not None:
            #    self._agent_position = position
            # if rotation is not None:
            #    self._agent_rotation = rotation

        return s_ if self.uenv.sapsc.success else None

    def set_agent_state(self, position: Optional[List[float]] = None,
                        rotation: Optional[List[float]] = None):
        """Set the agent position or rotation

        Args:
            position: a list of x,y,z in Unity's coordinate system
            rotation: a list of x,y,z,w in Unity's coordinate system

        If the position or rotation is not provided as argument,
        then it takes them from the current state.

        Returns:
            True if the state is set, else False

        """

        # Just added to cause exception
        pos, rot = self.agent_position, self.agent_rotation

        agent_id = 0
        state = [agent_id]
        state += pos if position is None else position
        state += rot if rotation is None else rotation

        unity_output = self.uenv.process_immediate_message(
            self.uenv.sapsc.build_immediate_request("agentPosition",
                                                    state))

        if self.uenv.sapsc.success:
            s_, _, _, _ = self._single_step(
                steps_from_proto(
                    unity_output.rl_output.agentInfos["immediate"].value,
                    self.uenv._env_specs[self.name]
                )[1]
            )
            self._set_obs(s_)
            # if position is not None:
            #    self._agent_position = position
            # if rotation is not None:
            #    self._agent_rotation = rotation

        return self.uenv.sapsc.success

    def set_agent_position(self, position: Optional[List[float]]):
        """Set the agent position

        Args:
            position: a list of x,y,z in Unity's coordinate system

        If the position is not provided as argument,
        then it takes them from the current state.

        Returns:
            True if the state is set, else False

        """

        return self.set_agent_state(position=position)

    def set_agent_rotation(self, rotation: Optional[List[float]]):
        """Set the agent  rotation

        Args:
            rotation: a list of x,y,z,w in Unity's coordinate system

        If the rotation is not provided as argument,
        then it takes them from the current state.

        Returns:
            True if the state is set, else False

        """

        return self.set_agent_state(rotation=rotation)

    # expose properties from uenv
    @property
    def actions(self):
        return self.uenv.actions

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

    def unity_to_navmap_location(self, unity_x, unity_z):
        """Convert a location from Unity's 3D coordinate system to navigable map's 2D coordinate system

        Args:
            unity_x: x coordinate in unity
            unity_z: z coordinate in unity

        Returns:
            navmap_x, navmap_y
        """
        # TODO: 0 <= unity_x < math.floor(unity_max_x) && 0 <= unity_z < math.floor(unity_max_z)
        navmap_x = math.floor(
            unity_x / (math.floor(self.uenv.unity_max_x) / self.uenv.navmap_max_x))
        navmap_y = math.floor(
            unity_z / (math.floor(self.uenv.unity_max_z) / self.uenv.navmap_max_y))
        return navmap_x, navmap_y

    def navmap_to_unity_location(self, navmap_x, navmap_y, navmap_cell_center=True):
        """Convert a location from navigable map's 2D coordinate system to Unity's 3D coordinate system

        Args:
            navmap_x, navmap_y: x, y location on navmap
            navmap_cell_center: Whether to return the point in cell center, default True.

        Returns:
            unity_x, unity_z
        """

        # TODO:  input: 0 <= navmap_x < navmap_max_x && 0<= navmap_y < navmap_max_y
        unity_x = navmap_x * (math.floor(self.uenv.unity_max_x) / self.uenv.navmap_max_x)
        unity_z = navmap_y * (math.floor(self.uenv.unity_max_z) / self.uenv.navmap_max_y)
        if navmap_cell_center:
            unity_x += (math.floor(self.uenv.unity_max_x) / self.uenv.navmap_max_x) / 2
            unity_z += (math.floor(self.uenv.unity_max_z) / self.uenv.navmap_max_y) / 2

        return unity_x, unity_z

    def unity_to_navmap_rotation(self, unity_rotation: List[float]):
        """Convert a rotation from Unity's quarternion to navigable map's 2D coordinate system

        Args:
            unity_rotation: [x,y,z,w] in unity quaternion system

        Returns:
            [x,y] vector components of rotation
        """
        x, _, z = _qv_mult(unity_rotation, [0, 0, 1])
        return list(_normalize(np.asarray([x, z])))

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

        v1 = _normalize(np.asarray([x, 0, y]))
        v2 = _normalize(np.cross([0.1, 0], v1))
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
        return self.uenv.sample_navigable_point(x=x, y=y, z=z)

    def is_navigable(self, x: float, y: float, z: float) -> bool:
        """Returns if the point is navigable or not

        Args:
            x,y,z: the unity coordinates of the point to check

        Returns:
            True if the point represented by x,y,z is navigable else False
        """
        return self.sample_navigable_point(x, y, z)

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

    @property
    def shortest_path_length(self):
        """the shortest navigable path length from current location to
        goal position
        """
        return self.uenv.shortest_path_length

    @property
    def shortest_path(self):
        return self.uenv.shortest_path
