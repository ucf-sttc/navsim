import random

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

import torch
import torch.nn.functional as F
from numpy.core._multiarray_umath import ndarray
from torch import nn
import numpy as np
from tqdm import tqdm

from copy import deepcopy

from typing import Union, Optional


class AirSimEnv:
    def __init__(self, filename: Optional[str] = None, observation_mode: int = 0):
        engine_side_channel = EngineConfigurationChannel()
        environment_side_channel = EnvironmentParametersChannel()
        uenv = UnityEnvironment(file_name=filename,
                                log_folder='/tmp/unity', seed=123,
                                side_channels=[engine_side_channel, environment_side_channel])
        engine_side_channel.set_configuration_parameters(time_scale=10, quality_level=0)
        environment_side_channel.set_float_parameter("rewardForGoalCollision", .5)
        environment_side_channel.set_float_parameter("rewardForExplorationPointCollision", .005)
        environment_side_channel.set_float_parameter("rewardForOtherCollision", -.1)
        environment_side_channel.set_float_parameter("rewardForFallingOffMap", -1)
        environment_side_channel.set_float_parameter("rewardForEachStep", -.0001)
        environment_side_channel.set_float_parameter("segmentationMode", 1)
        environment_side_channel.set_float_parameter("observationMode", observation_mode)

        genv = UnityToGymWrapper(uenv, False, False,
                                 True)  # (Environment, uint8_visual, flatten_branched, allow_multiple_obs)
        genv.action_space.np_random.seed(123)
        self.uenv = uenv
        self.genv = genv
        self.observation_mode = observation_mode
        self.n_x = 7
        self.n_y = 2

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


class Memory:
    """
    Stores and returns list of numpy arrays for s a r s_ d

    assumes you are storing s a r s_ d : state action reward next_state done
    # state / next_state : tuple of n-dim numpy arrays
    # a : 1-d numpy array of floats or ints
    # r : 1-d numpy array of floats #TODO: rewards could also be n-dim
    # d : 1-d numpy array of booleans

    # TODO: see if sampling can be made better with np.random
    # TODO: if we made this tensor based instead of np, would it make it better ?
    """

    @classmethod
    def sample_info(self, s, a, r, s_, d):
        print("Sample Info")
        print('-----------')
        print("Shapes:")
        print('s:', [a.shape for a in s])
        print('a:', a.shape)
        print('r:', r.shape)
        print('s_:', [a.shape for a in s_])
        print('d:', d.shape)

    def __init__(self, capacity: int,
                 state_shapes: Union[list, tuple, int],
                 action_shape: Union[list, tuple, int],
                 seed: Optional[float] = None):
        """

        :param capacity: total capacity of the memory
        :param state_shapes: a tuple of state dimensions
        :param action_shape: int representing number of actions
        :param seed: seed to provide to numpy
        """
        self.capacity = capacity
        self.ptr: int = 0
        self.size: int = 0

        self.seed: Optional[float] = seed
        # from numpy.random import MT19937, RandomState, Generator
        # mt = MT19937(seed)
        # rs = RandomState(mt)
        # rg = Generator(mt)
        self.rng = np.random.default_rng(seed)

        #if isinstance(state_shapes, (list, tuple)):  # state_dims is an int
        if isinstance(state_shapes, (int,float)):  # state_dims is an int
            state_shapes = [[state_shapes]]  # make state_dims a list of sequences
        elif isinstance(state_shapes[0], (int, float)):  # state_dims first member is an int
            state_shapes = [state_shapes]  # make state_dims a list of sequences
        # else state_dims is a list of sequences, i.e. [[state_shape], [state_shap], [state_shape]]

        s_len = len(state_shapes)
        self.s = [None] * s_len
        self.s_ = [None] * s_len

        for i in range(s_len):
            state_shapes[i] = [capacity] + list(state_shapes[i])
            self.s[i] = np.full(state_shapes[i], 0.0)
            self.s_[i] = np.full(state_shapes[i], 0.0)

        action_shape = [capacity] + list(action_shape)
        self.a = np.full(action_shape, 0.0)

        self.r = np.full((capacity, 1), 0.0)
        self.d = np.full((capacity, 1), False)

    def append(self, s: Union[list, tuple, int, float],
               a: Union[list, tuple, int, float],
               r: float,
               s_: Union[list, tuple, int, float],
               d: int):
        if isinstance(s, (int, float)):  # state is an int
            s = [[s]]  # make state a list of sequences
            s_ = [[s_]]
        elif isinstance(s[0], (int,float)):  # state does not have multiple seq
            s = [s]
            s_ = [s_]
        for i in range(len(s)):
            self.s[i][self.ptr] = s[i]
            self.s_[i][self.ptr] = s_[i]

        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.d[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, size):
        if (size == 0) or (size >= self.size):
            # idx = list(range(self.size))   # just return all
            return (
                self.s,
                self.a,
                self.r,
                self.s,
                self.d
            )
        else:
            idx = self.rng.integers(low=0, high=self.size, size=size)
            s_len = len(self.s)
            return (
                [self.s[i][idx] for i in range(s_len)],
                self.a[idx],
                self.r[idx],
                [self.s_[i][idx] for i in range(s_len)],
                self.d[idx]
            )

    def info(self):
        print("Memory Info")
        print('-----------')
        print('capacity:', self.capacity)
        print('size:', self.size)
        print('seed:', self.seed)
        print("Shapes:")
        print('s:', [a.shape for a in self.s])
        print('a:', self.a.shape)
        print('r:', self.r.shape)
        print('s_:', [a.shape for a in self.s_])
        print('d:', self.d.shape)

    def __len__(self):
        return len(self.mem)

    def __repr__(self):
        return str(self.mem)

    def __str__(self):
        return f'capacity={self.capacity} \nlength={len(self.mem)} \ncurrent pointer={self.ptr} \nmemory contents:\n{self.mem}'

    def __getitem__(self, sliced):
        return Memory(capacity=self.capacity, seed=self.seed, mem=self.mem[sliced])

    def pretty_print(self, n_rows=0):
        """

        :param n_rows: 0 means all
        :return:
        """
        np.set_printoptions(precision=2)
        n_rows = n_rows or len(self.mem)
        for i in range(n_rows):
            for e in self.mem[i]:
                print(e, end=' ')
            print


# for actor:
#   n_x is state dimension
#   n_y is action dimension
#   output: max_action * tanh(fc3)
#   returns a
# for critic:
#   n_x is state_dimension + action_dimension
#   n_y is 1
#   input:  cat([state,action],1)
#   returns q

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


class ModelTorch(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf

        self.fc1 = nn.Linear(self.conf.n_x, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self.conf.n_y)

    def forward(self, x):
        """

        :param x: numpy array
        :return:
        """
        x = torch.FloatTensor(x).to(self.device),
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AirSimNN(object):
    def __init__(self, conf):
        self.conf = conf
        self.model = ModelTorch(conf)
        self.opt_fn = torch.optim.RMSprop(self.model.parameters())
        self.los_fn = torch.nn.MSELoss()
        self.device = torch.device('cpu')  # 'cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        train_generator = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(x_train).type(torch.Tensor).view([-1, self.conf.n_x]),
                torch.from_numpy(y_train).type(torch.Tensor).view([-1, self.conf.n_y])
            ), batch_size=self.conf.n_batch_size, shuffle=True
        )
        for epoch in range(self.conf.n_epochs):
            for x_batch, y_batch in train_generator:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                y_pred = self.model(x_batch)
                loss = self.los_fn(y_pred, y_batch)
                self.opt_fn.zero_grad()
                loss.backward(retain_graph=True)
                self.opt_fn.step()
        return self

    def predict(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x).type(torch.Tensor).view([-1, self.conf.n_x]).to(self.device)
            y_pred = self.model(x)
            y_pred = y_pred.detach().numpy()
        return y_pred

    def info(self):
        print(self.conf)
        print(self.model)


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
        # state = state[0]
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class DDPGAgent(object):

    def __init__(self, env, device, discount=0.99, tau=0.005):
        self.env = env
        self.device = device
        self.discount = discount
        self.tau = tau
        # self.state_dim = state_dim

        self.max_action = self.env.action_space.high[0]

        print(self.env.observation_space_shapes[0])
        if self.env.observation_mode == 0:
            self.vector_state_dimension = self.env.observation_space_shapes[0][0]
        elif self.env.observation_mode == 1:
            self.vector_state_dimension = None
        elif self.env.observation_mode == 2:
            self.vector_state_dimension = self.env.observation_space_shapes[3][0]

        if (self.env.observation_mode == 0) or (self.env.observation_mode == 2):
            self.actor = ActorVector(self.vector_state_dimension, self.env.action_space_shape[0], self.max_action).to(device)
            self.critic = CriticVector(self.vector_state_dimension, self.env.action_space_shape[0]).to(device)
        else:  # TODO: Implement VisualActor here
            self.actor = None
            self.critic = None

        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    def select_action(self, state):
        if self.env.observation_mode == 0:
            state = torch.FloatTensor(state[0].reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    @staticmethod
    def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_checkpoint(self, filename):
        torch.save(self.critic.state_dict(), filename + '_critic')
        torch.save(self.critic_optimizer.state_dict(), filename + '_critic_optimizer')
        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.actor_optimizer.state_dict(), filename + '_actor_optimizer')

    def save_actor(self, filename):
        dummy_input = torch.randn(self.state_dim).to(self.device)
        torch.onnx.export(self.actor, dummy_input, filename, export_params=True, opset_version=9,
                          do_constant_folding=True,
                          input_names=['state'], output_names=['action'])

    def load_checkpoint(self, filename):
        self.critic.load_state_dict(
            torch.load(
                filename + "_critic",
                map_location=torch.device('cpu')
            )
        )
        self.critic_optimizer.load_state_dict(
            torch.load(
                filename + "_critic_optimizer",
                map_location=torch.device('cpu')
            )
        )
        self.critic_target = deepcopy(self.critic)
        self.actor.load_state_dict(
            torch.load(
                filename + "_actor",
                map_location=torch.device('cpu')
            )
        )
        self.actor_optimizer.load_state_dict(
            torch.load(
                filename + "_actor_optimizer",
                map_location=torch.device('cpu')
            )
        )
        self.actor_target = deepcopy(self.actor)

    def train(self, memory, batch_size=100):
        state, action, next_state, reward, episode_done = memory.sample(batch_size)
        if self.env.observation_mode == 0:
            state = torch.FloatTensor(state[0]).to(self.device)
            next_state = torch.FloatTensor(next_state[0]).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        episode_done = torch.FloatTensor(episode_done).to(self.device)

        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + ((1.0 - episode_done) * self.discount * target_q).detach()
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        DDPGAgent.soft_update(self.critic, self.critic_target, self.tau)
        DDPGAgent.soft_update(self.actor, self.actor_target, self.tau)


def evaluate_policy(policy, env, seed, eval_episodes=10, render=False):
    eval_env = env
    eval_env.seed(seed + 100)
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(state)
            if render:
                eval_env.render()
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


class Trainer:

    def __init__(self, config, env, enable_logging=False):
        """

        :param config: A DictObj containing dictionary and object interface
        :param env:
        :param enable_logging:
        """
        self.enable_logging = enable_logging
        self.config = config
        self.env = env

        self.max_action = self.env.action_space.high[0]


        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.agent = DDPGAgent(
            env=self.env,
            device=self.device,
            discount=self.config['discount'], tau=self.config['tau']
        )

        # TODO: Give file and folder names better names based on the experiment_id

        self.save_file_name = f"DDPG_{self.config['env_name']}_{self.config['seed']}"
        self.memory = Memory(
            capacity=self.config.memory_capacity,
            state_shapes=self.env.observation_space_shapes,
            action_shape=self.env.action_space_shape,
            seed=self.config['seed']
        )  # ReplayBuffer(self.vector_state_dimension, self.action_dimension)

        self.apply_seed()

        if self.enable_logging:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter('./logs/' + self.config['env_name'] + '/')
        try:
            os.mkdir('./models')
        except Exception as e:
            pass

    def apply_seed(self):
        self.env.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])

    def train(self):  # at three places state has been selected as 0th element
        state = self.env.reset()
        episode_done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        evaluations = []
        episode_rewards = []
        for ts in tqdm(range(1, int(self.config['time_steps']) + 1)):
            episode_timesteps += 1
            if ts < self.config['start_time_step']:
                action = self.env.action_space.sample()
            else:
                action = (
                        self.agent.select_action(state) + np.random.normal(
                    0, self.max_action * self.config['expl_noise'],
                    size=self.env.action_shape[0]
                )
                ).clip(
                    -self.max_action,
                    self.max_action
                )
            next_state, reward, episode_done, _ = self.env.step(action)
            #next_state = next_state[0]

            self.memory.append(
                s=state, a=action, s_=next_state, r=reward,
                d = float(episode_done) if episode_timesteps < self.config.max_episode_steps else 1)
            state = next_state
            episode_reward += reward
            if ts >= self.config['start_time_step']:
                self.agent.train(self.memory, self.config['batch_size'])
            if episode_done:
                if self.enable_logging:
                    self.writer.add_scalar('Episode Reward', episode_reward, ts)
                episode_rewards.append(episode_reward)
                state = self.env.reset()[0]
                episode_done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
        if ts % 1000 == 0:
            evaluations.append(evaluate_policy(self.agent, self.config['env_name'], self.config['seed']))
            self.agent.save_checkpoint(f"./models/{self.save_file_name}")
        return episode_rewards, evaluations
