class ReplayBuffer(object):

    def __init__(self, state_dimension, action_dimension, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dimension))
        self.action = np.zeros((max_size, action_dimension))
        self.next_state = np.zeros((max_size, state_dimension))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        """
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
        """
        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind]
        )



class ActorVector(torch.nn.Module):

    def __init__(self, state_dimension, action_dimension, max_action):
        super(ActorVector, self).__init__()
        self.l1 = torch.nn.Linear(state_dimension, 400)
        self.l2 = torch.nn.Linear(400, 300)
        self.l3 = torch.nn.Linear(300, action_dimension)
        self.max_action = max_action

    def forward(self, state):
        """

        :param state: state is a torch tensor
        :return:
        """
        # curframe = inspect.currentframe()
        # calframe = inspect.getouterframes(curframe, 2)
        # print('actor caller name:', calframe[1][3])
        # print('actor',state.shape)
        # print('actor',self.l1)
        # state = state[0]
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return self.max_action * a

class CriticVector(torch.nn.Module):

    def __init__(self, state_dimension, action_dimension):
        super(CriticVector, self).__init__()
        self.l1 = torch.nn.Linear(state_dimension + action_dimension, 400)
        self.l2 = torch.nn.Linear(400, 300)
        self.l3 = torch.nn.Linear(300, 1)

    def forward(self, state, action):
        """

        :param state: a torch Tensor
        :param action: a torch Tensor
        :return:
        """
        # print('critic',state.shape)
        # print('critic',self.l1)

        # state = state[0]
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q

class NavSimEnv:
    def __init__(self, conf):
        """
        conf: ObjDict having Environment Conf
        :param conf:
        """
        # filename: Optional[str] = None, obs_mode: int = 0, max_steps:int = 5):
        self.conf = conf

        self.observation_mode = self.conf['obs_mode']
        self.uenv = None
        self.genv = None

        self.env_open = False
        self.open()

    def open(self):

        if self.env_open:
            raise ValueError('Environment already open')
        else:
            log_folder = Path(self.conf['log_folder'])
            log_folder.mkdir(parents=True, exist_ok=True)

            engine_side_channel = EngineConfigurationChannel()
            environment_side_channel = EnvironmentParametersChannel()

            engine_side_channel.set_configuration_parameters(time_scale=10, quality_level=0)
            environment_side_channel.set_float_parameter("rewardForGoalCollision", self.conf['reward_for_goal'])
            environment_side_channel.set_float_parameter("rewardForExplorationPointCollision",
                                                         self.conf['reward_for_ep'])
            environment_side_channel.set_float_parameter("rewardForOtherCollision", self.conf['reward_for_other'])
            environment_side_channel.set_float_parameter("rewardForFallingOffMap",
                                                         self.conf['reward_for_falling_off_map'])
            environment_side_channel.set_float_parameter("rewardForEachStep", self.conf['reward_for_step'])
            environment_side_channel.set_float_parameter("segmentationMode", self.conf['segmentation_mode'])
            environment_side_channel.set_float_parameter("observationMode", self.conf['obs_mode'])
            environment_side_channel.set_float_parameter("episodeLength", self.conf['max_steps'])
            environment_side_channel.set_float_parameter("selectedTaskIndex", self.conf['task'])
            environment_side_channel.set_float_parameter("goalSelectionIndex", self.conf['goal'])
            environment_side_channel.set_float_parameter("agentCarPhysics", self.conf['agent_car_physics'])

            uenv_file_name = str(Path(self.conf['env_path']).resolve()) if self.conf['env_path'] else None
            self.uenv = UnityEnvironment(file_name=uenv_file_name,
                                         log_folder=str(log_folder.resolve()),
                                         seed=self.conf['seed'],
                                         timeout_wait=self.conf['timeout'],
                                         worker_id=self.conf['worker_id'],
                                         # base_port=self.conf['base_port'],
                                         no_graphics=False,
                                         side_channels=[engine_side_channel, environment_side_channel])

            self.genv = UnityToGymWrapper(self.uenv, False, False, True)
            # (Env, uint8_visual, flatten_branched, allow_multiple_obs)
            self.seed(self.conf['seed'])
            self.env_open = True

    def close(self):
        if self.env_open:
            self.env_open = False
            if self.uenv is None:
                print('uenv is None')
            else:
                self.uenv.close()
        else:
            raise ValueError('Environment not open')

    def info(self):
        print('-----------')
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
        print('Self Observation Space Shapes:', self.observation_space_shapes)
        print('Self Observation Space Types:', self.observation_space_types)
        print('Reward Range:', self.genv.reward_range)
        print('Metadata:', self.genv.metadata)

    def info_steps(self):
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
        return [obs.shape for obs in self.observation_space.spaces]

    @property
    def observation_space_types(self):
        """
        Returns the dimensions of the observation space
        :return:
        """
        return [type(obs) for obs in self.observation_space.spaces]

    @property
    def gym_env(self):
        return self.genv

    @property
    def unity_env(self):
        return self.uenv

    def step(self, action):
        return self.genv.step(action)

    def reset(self):
        return self.genv.reset()

    def seed(self, seed=None):
        # TODO: This is broken in gym 0.10.9, fix it when they fix it
        # self.genv.action_space.seed(seed)
        # self.genv.seed(seed)
        pass
