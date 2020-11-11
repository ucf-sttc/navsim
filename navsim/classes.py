import csv
import os
import pickle
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

from copy import deepcopy



from typing import Union, Optional

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

import inspect

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
        #curframe = inspect.currentframe()
        #calframe = inspect.getouterframes(curframe, 2)
        #print('actor caller name:', calframe[1][3])
        #print('actor',state.shape)
        #print('actor',self.l1)
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
        #print('critic',state.shape)
        #print('critic',self.l1)

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
        state = {
            "critic":self.critic.state_dict(),
            "critic_optimizer":self.critic_optimizer.state_dict(),
            "actor":self.actir.state_dict(),
            "actor_optimizer":self.actor_optimizer.state_dict()

        }
        torch.save(state,filename)
        """
        torch.save(self.critic.state_dict(), filename + '_critic')
        torch.save(self.critic_optimizer.state_dict(), filename + '_critic_optimizer')
        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.actor_optimizer.state_dict(), filename + '_actor_optimizer')
        """
    def save_actor(self, filename):
        model = self.actor
        l1_shape = list(model.parameters())[0].shape #shape_of_first_layer
        device = next(model.parameters()).device
        #print(shape_of_first_layer)
        dummy_input = torch.randn(l1_shape[1:]).to(device)
        torch.onnx.export(model, dummy_input, filename, export_params=True, opset_version=9,
                          do_constant_folding=True,
                          input_names=['state'], output_names=['action'])
        """
        if (self.env.observation_mode == 0) or (self.env.observation_mode == 2):
            dummy_input = torch.randn(self.vector_state_dimension).to(self.device)
            torch.onnx.export(self.actor, dummy_input, filename, export_params=True, opset_version=9,
                          do_constant_folding=True,
                          input_names=['state'], output_names=['action'])
        """
    def load_checkpoint(self, filename):
        state = torch.load(filename)
        self.critic.load_state_dict(state['critic'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.actor.load_state_dict(state['actor'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])

        """
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
        """
        self.actor_target = deepcopy(self.actor)

    def train(self, state, action, reward, next_state, episode_done):
        #print('agent train state ',state.shape)
        #print('agent train next_state',next_state.shape)
        if self.env.observation_mode == 0:
            state = torch.FloatTensor(state[0]).to(self.device)
            next_state = torch.FloatTensor(next_state[0]).to(self.device)
        #print('agent train',next_state.shape)
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