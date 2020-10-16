from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel


class AirSimEnv:
    def __init__(self, conf):  # filename: Optional[str] = None, observation_mode: int = 0, max_steps:int = 5):
        self.conf = conf
        engine_side_channel = EngineConfigurationChannel()
        environment_side_channel = EnvironmentParametersChannel()
        self.uenv = UnityEnvironment(file_name=self.conf['filename'],
                                     #log_folder=self.conf['log_folder'],
                                     seed=self.conf['seed'],
                                     side_channels=[engine_side_channel, environment_side_channel])
        engine_side_channel.set_configuration_parameters(time_scale=10, quality_level=0)
        environment_side_channel.set_float_parameter("rewardForGoalCollision", self.conf['reward_for_goal'])
        environment_side_channel.set_float_parameter("rewardForExplorationPointCollision", self.conf['reward_for_ep'])
        environment_side_channel.set_float_parameter("rewardForOtherCollision", self.conf['reward_for_other'])
        environment_side_channel.set_float_parameter("rewardForFallingOffMap", self.conf['reward_for_falling_off_map'])
        environment_side_channel.set_float_parameter("rewardForEachStep", self.conf['reward_for_step'])
        environment_side_channel.set_float_parameter("segmentationMode", self.conf['segmentation_mode'])
        environment_side_channel.set_float_parameter("observationMode", self.conf['observation_mode'])
        environment_side_channel.set_float_parameter("episodeLength", self.conf['max_steps'])

        self.genv = UnityToGymWrapper(self.uenv, False, False, True)
        # (Env, uint8_visual, flatten_branched, allow_multiple_obs)
        self.genv.action_space.np_random.seed(123)
        self.observation_mode = self.conf['observation_mode']

    def close(self):
        self.uenv.close()

    def info(self):
        print("Env Info")
        print('-----------')
        if self.genv.spec:
            print(self.genv.spec.id)
        print('Action Space:', self.action_space)
        print('Action sample:', self.action_space.sample())
        print('Action Space Shape:', self.action_space.shape)
        print('Action Space Low:', self.action_space.low)
        print('Action Space High:', self.action_space.high)
        print('Observation Mode:', self.observation_mode)
        print('Gym Observation Space:', self.genv.observation_space)
        print('Gym Observation Space Shape:', self.genv.observation_space.shape)
        print('Self Observation Space:', self.observation_space)
        print('Self Observation Space Shape:', self.observation_space.shape)
        print('Reward Range:', self.genv.reward_range)
        print('Metadata:', self.genv.metadata)
        print('Initial State:', self.reset())
        print('First Step State:', self.step(self.action_space.sample()))

    @property
    def action_space(self):
        return self.genv.action_space

    @property
    def action_space_shape(self):
        return self.action_space.shape

    @property
    def observation_space(self):
        """
        We can control here what observation space to return
        Observation Space is always returned as a tuple
        :return:
        """
        return self.genv.observation_space

    @property
    def observation_space_shapes(self):
        """
        Returns the dimensions of the observation space
        :return:
        """
        return [obs.shape for obs in self.observation_space]

    @property
    def gym_env(self):
        return self.genv

    @property
    def unity_env(self):
        return self.uenv

    def step(self, a):
        return self.genv.step(a)

    def reset(self):
        return self.genv.reset()

    def seed(self, seed=None):
        return self.genv.seed(seed)
