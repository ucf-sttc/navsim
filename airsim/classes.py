from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from tqdm import tqdm

from copy import deepcopy


class AirSimEnv:
    def __init__(self, filename=None):
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

        genv = UnityToGymWrapper(uenv, False, False,
                                 True)  # (Environment, uint8_visual, flatten_branched, allow_multiple_obs)
        genv.action_space.np_random.seed(123)
        self.uenv = uenv
        self.genv = genv
        self.n_x = 7
        self.n_y = 2

    def close(self):
        self.uenv.close()

    def info(self):
        genv = self.genv
        if genv.spec:
            print(genv.spec.id)
        print('Action Space:', genv.action_space)
        print('Action sample:', genv.action_space.sample())
        print('Action Space Shape:', genv.action_space.shape)
        print('Action Space Low:', genv.action_space.low)
        print('Action Space High:', genv.action_space.high)
        print('Gym Observation Space:', genv.observation_space)
        print('Gym Observation Space Shape:', genv.observation_space.shape)
        print('Self Observation Space:', self.observation_space)
        print('Self Observation Space Shape:', self.observation_space.shape)
        print('Reward Range:', genv.reward_range)
        print('Metadata:', genv.metadata)
        print('Initial State:', genv.reset())
        print('First Step State:', genv.step(genv.action_space.sample()))

    @property
    def action_space(self):
        return self.genv.action_space

    @property
    def observation_space(self):
        """
        We can control here what observation space to return
        :return:
        """
        return self.genv.observation_space[0]

    @property
    def gym_env(self):
        return self.genv

    def step(self, a):
        return self.genv.step(a)

    def reset(self):
        return self.genv.reset()

    def seed(self, seed=None):
        return self.genv.seed(seed)


class Memory():
    """
    Stores and returns list of numpy arrays
    """

    def __init__(self, capacity, seed=None, mem=[]):
        self.cap = capacity
        self.mem = mem
        self.ptr = 0
        self.seed = seed
        seed or random.seed(seed)

    def append(self, item):
        if len(self.mem) < self.cap:
            self.mem.append(None)
        self.mem[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.cap

    def sample(self, size):
        # TODO: see if it can be made better with numpy
        if (size == 0) or (size >= len(self.mem)):
            return self.mem  # just return all
        else:
            return random.sample(self.mem, size)

    def __len__(self):
        return len(self.mem)

    def __repr__(self):
        return str(self.mem)

    def __str__(self):
        return f'capacity={self.cap} \nlength={len(self.mem)} \ncurrent pointer={self.ptr} \nmemory contents:\n{self.mem}'

    def __getitem__(self, sliced):
        return Memory(capacity=self.cap, seed=self.seed, mem=self.mem[sliced])

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
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
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


class Actor(torch.nn.Module):

    def __init__(self, state_dimension, action_dimension, max_action):
        super(Actor, self).__init__()
        self.l1 = torch.nn.Linear(state_dimension, 400)
        self.l2 = torch.nn.Linear(400, 300)
        self.l3 = torch.nn.Linear(300, action_dimension)
        self.max_action = max_action

    def forward(self, state):
        #state = state[0]
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(torch.nn.Module):

    def __init__(self, state_dimension, action_dimension):
        super(Critic, self).__init__()
        self.l1 = torch.nn.Linear(state_dimension + action_dimension, 400)
        self.l2 = torch.nn.Linear(400, 300)
        self.l3 = torch.nn.Linear(300, 1)

    def forward(self, state, action):
        #state = state[0]
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)


class DDPGAgent(object):

    def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.state_dim = state_dim
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
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
        torch.onnx.export(self.actor,dummy_input,filename,export_params=True,opset_version=9,do_constant_folding=True,
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

    def train(self, replay_buffer, batch_size=100):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + (not_done * self.discount * target_q).detach()
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
        state, done = eval_env.reset()[0], False
        while not done:
            action = policy.select_action(np.array(state))
            if render:
                eval_env.render()
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


class Trainer:

    def __init__(self, conf, env, enable_logging=False):
        self.enable_logging = enable_logging
        self.config = conf
        self.env = env
        self.apply_seed()
        self.state_dimension = self.env.observation_space.shape[0]
        self.action_dimension = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.agent = DDPGAgent(
            state_dim=self.state_dimension, action_dim=self.action_dimension,
            max_action=self.max_action, device=self.device,
            discount=self.config['discount'], tau=self.config['tau']
        )
        self.save_file_name = f"DDPG_{self.config['env_name']}_{self.config['seed']}"
        self.memory = ReplayBuffer(self.state_dimension, self.action_dimension)
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
        state = self.env.reset()[0]
        done = False
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
                        self.agent.select_action(np.array(state)) + np.random.normal(
                    0, self.max_action * self.config['expl_noise'],
                    size=self.action_dimension
                )
                ).clip(
                    -self.max_action,
                    self.max_action
                )
            next_state, reward, done, _ = self.env.step(action)
            next_state = next_state[0]

            self.memory.add(
                state, action, next_state, reward,
                float(done) if episode_timesteps < self.config.max_episode_steps else 0)
            state = next_state
            episode_reward += reward
            if ts >= self.config['start_time_step']:
                self.agent.train(self.memory, self.config['batch_size'])
            if done:
                if self.enable_logging:
                    self.writer.add_scalar('Episode Reward', episode_reward, ts)
                episode_rewards.append(episode_reward)
                state = self.env.reset()[0]
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
        if ts % 1000 == 0:
            evaluations.append(evaluate_policy(self.agent, self.config['env_name'], self.config['seed']))
            self.agent.save_checkpoint(f"./models/{self.save_file_name}")
        return episode_rewards, evaluations
