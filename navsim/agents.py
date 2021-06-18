import torch
import torch.nn.functional as F
from torch import nn

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
        self.device = torch.device(
            'cpu')  # 'cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        train_generator = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(x_train).type(torch.Tensor).view(
                    [-1, self.conf.n_x]),
                torch.from_numpy(y_train).type(torch.Tensor).view(
                    [-1, self.conf.n_y])
            ), batch_size=self.conf.n_batch_size, shuffle=True
        )
        for epoch in range(self.conf.n_epochs):
            for x_batch, y_batch in train_generator:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(
                    self.device)
                y_pred = self.model(x_batch)
                loss = self.los_fn(y_pred, y_batch)
                self.opt_fn.zero_grad()
                loss.backward(retain_graph=True)
                self.opt_fn.step()
        return self

    def predict(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x).type(torch.Tensor).view(
                [-1, self.conf.n_x]).to(self.device)
            y_pred = self.model(x)
            y_pred = y_pred.detach().numpy()
        return y_pred

    def info(self):
        print(self.conf)
        print(self.model)


import math


def num2tuple(num):
    return tuple(num) if isinstance(num, (tuple, list)) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    h_w, kernel_size, stride, pad, dilation = num2tuple(h_w), \
                                              num2tuple(kernel_size), \
                                              num2tuple(stride), \
                                              num2tuple(pad), \
                                              num2tuple(dilation)
    pad = num2tuple(pad[0]), num2tuple(pad[1])
    # print(h_w,kernel_size,stride,pad,dilation)
    h = math.floor(
        (h_w[0] + sum(pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) /
        stride[0] + 1)
    w = math.floor(
        (h_w[1] + sum(pad[1]) - dilation[1] * (kernel_size[1] - 1) - 1) /
        stride[1] + 1)

    return h, w


def convtransp2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1,
                              out_pad=0):
    h_w, kernel_size, stride, pad, dilation, out_pad = num2tuple(h_w), \
                                                       num2tuple(
                                                           kernel_size), num2tuple(
        stride), num2tuple(
        pad), num2tuple(dilation), num2tuple(out_pad)
    pad = num2tuple(pad[0]), num2tuple(pad[1])

    h = (h_w[0] - 1) * stride[0] - sum(pad[0]) + dilation[0] * (
            kernel_size[0] - 1) + out_pad[0] + 1
    w = (h_w[1] - 1) * stride[1] - sum(pad[1]) + dilation[1] * (
            kernel_size[1] - 1) + out_pad[1] + 1

    return h, w


def conv2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1):
    h_w_in, h_w_out, kernel_size, stride, dilation = num2tuple(
        h_w_in), num2tuple(h_w_out), \
                                                     num2tuple(
                                                         kernel_size), num2tuple(
        stride), num2tuple(dilation)

    p_h = ((h_w_out[0] - 1) * stride[0] - h_w_in[0] + dilation[0] * (
            kernel_size[0] - 1) + 1)
    p_w = ((h_w_out[1] - 1) * stride[1] - h_w_in[1] + dilation[1] * (
            kernel_size[1] - 1) + 1)

    return (math.floor(p_h / 2), math.ceil(p_h / 2)), (
        math.floor(p_w / 2), math.ceil(p_w / 2))


def convtransp2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1,
                             dilation=1, out_pad=0):
    h_w_in, h_w_out, kernel_size, stride, dilation, out_pad = num2tuple(
        h_w_in), num2tuple(h_w_out), \
                                                              num2tuple(
                                                                  kernel_size), num2tuple(
        stride), num2tuple(
        dilation), num2tuple(out_pad)

    p_h = -(h_w_out[0] - 1 - out_pad[0] - dilation[0] * (kernel_size[0] - 1) - (
            h_w[0] - 1) * stride[0]) / 2
    p_w = -(h_w_out[1] - 1 - out_pad[1] - dilation[1] * (kernel_size[1] - 1) - (
            h_w[1] - 1) * stride[1]) / 2

    return (math.floor(p_h / 2), math.ceil(p_h / 2)), (
        math.floor(p_w / 2), math.ceil(p_w / 2))


class Actor(torch.nn.Module):
    """
    state_dimensions: Should always be a list of state_dimension
                      if vector is True then vector obs should always be last
                      if visual is True then visual obs should come before vec dimension
    """

    def __init__(self, state_dimensions, action_dimension, max_action):
        super(Actor, self).__init__()

        # All feature_layers end up with ReLU output
        self.feature_layers = torch.nn.ModuleList()
        l_out_size = []

        # visual network settings
        stride = 1
        padding = 0
        dilation = 1
        kernel_size = 2
        out_channels = 4

        for state_dim in state_dimensions:
            if len(state_dim) == 1:  # means we have vector observation
                out_size = state_dim[0] * 2  # make sure its always int
                layer = torch.nn.Sequential(
                    torch.nn.Linear(state_dim[0], out_size),
                    torch.nn.ReLU()
                    #                torch.nn.Linear(400, 300),
                    #                torch.nn.ReLU()
                )
                self.feature_layers.append(layer)
                l_out_size.append(out_size)
            else:  # visual:
                layer = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=state_dim[2],
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation),
                    # [batch_size, n_features_conv, height, width]
                    torch.nn.MaxPool2d(kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation),
                    torch.nn.ReLU()
                )
                h, w = conv2d_output_shape(h_w=state_dim[0:2],
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           pad=padding,
                                           dilation=dilation)
                h, w = conv2d_output_shape(h_w=[h, w],
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           pad=padding,
                                           dilation=dilation)
                l_out_size.append(h * w * out_channels)
                self.feature_layers.append(layer)

        #        self.l3 = torch.nn.Linear(300, action_dimension)

        cat_dim = sum(l_out_size)

        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(cat_dim, action_dimension),
            torch.nn.Tanh()
        )
        # print(f'{l_out_size},cat_dim:{cat_dim},action_dim:{action_dimension}')
        # self.action_out = torch.nn.Linear(cat_dim, action_dimension)

        self.max_action = max_action

    def forward(self, state):
        """

        :param state: state is a list of torch tensors
        :return:
        """
        # curframe = inspect.currentframe()
        # calframe = inspect.getouterframes(curframe, 2)
        # print('actor caller name:', calframe[1][3])
        # print('actor',state.shape)
        # print('actor',self.l1)
        # state = state[0]
        features = []
        for s, layer in zip(state, self.feature_layers):
            layer = layer(s)
            if layer.dim() > 2:
                layer = layer.view(layer.size(0), -1)
            features.append(layer)
            # print(layer.shape)

        #        if len(features) > 1:
        #            features_cat = torch.cat(features, dim=1)
        #        else:
        #            features_

        features_cat = torch.cat(features, dim=1) if len(features) > 1 else \
            features[0]
        a = self.out_layer(features_cat)
        return self.max_action * a


class Critic(torch.nn.Module):
    """
    TODO:
    Should we combine the state with action after some layers or right at the first layer?
    """

    def __init__(self, state_dimensions, action_dimension):
        super(Critic, self).__init__()

        # All feature_layers end up with ReLU output
        self.feature_layers = torch.nn.ModuleList()
        l_out_size = []

        # visual network settings
        stride = 1
        padding = 0
        dilation = 1
        kernel_size = 2
        out_channels = 4

        for state_dim in state_dimensions:
            if len(state_dim) == 1:  # means we have vector observation
                out_size = state_dim[0] * 2  # make sure its always int
                layer = torch.nn.Sequential(
                    torch.nn.Linear(state_dim[0], out_size),
                    torch.nn.ReLU()
                    #                torch.nn.Linear(400, 300),
                    #                torch.nn.ReLU()
                )
                self.feature_layers.append(layer)
                l_out_size.append(out_size)
            else:  # visual:
                layer = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=state_dim[2],
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation),
                    # [batch_size, n_features_conv, height, width]
                    torch.nn.MaxPool2d(kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation),
                    torch.nn.ReLU()
                )
                h, w = conv2d_output_shape(h_w=state_dim[0:2],
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           pad=padding,
                                           dilation=dilation)
                h, w = conv2d_output_shape(h_w=[h, w],
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           pad=padding,
                                           dilation=dilation)
                l_out_size.append(h * w * out_channels)
                self.feature_layers.append(layer)

        # add action also as one of the feature layers
        out_size = action_dimension  # make sure its always int
        layer = torch.nn.Sequential(
            torch.nn.Linear(action_dimension, out_size),
            torch.nn.ReLU()
            #                torch.nn.Linear(400, 300),
            #                torch.nn.ReLU()
        )
        self.feature_layers.append(layer)
        l_out_size.append(out_size)

        cat_dim = sum(l_out_size)

        self.out_layer = torch.nn.Sequential(
            torch.nn.Linear(cat_dim, 1),
        )

    def forward(self, state, action):
        """

        :param state: a list of torch Tensors
        :param action: a torch Tensor
        :return:
        """
        # print('critic',state.shape)
        # print('critic',self.l1)

        # state = state[0]

        features = []
        for s, layer in zip(state, self.feature_layers):
            layer = layer(s)
            if layer.dim() > 2:
                layer = layer.view(layer.size(0), -1)
            features.append(layer)
            # print(layer.shape)

        layer = self.feature_layers[-1]
        layer = layer(action)
        if layer.dim() > 2:
            layer = layer.view(layer.size(0), -1)
        features.append(layer)

        features_cat = torch.cat(features, dim=1) if len(features) > 1 else \
            features[0]

        q = self.out_layer(features_cat)
        return q


class DDPGAgent(object):

    def __init__(self, env, device, discount=0.99, tau=0.005):
        self.env = env
        self.device = device
        self.discount = discount
        self.tau = tau
        # self.state_dim = state_dim

        self.max_action = self.env.action_space.high[0]

        # if self.env.observation_mode == 0:
        #    self.vector_state_dimension = self.env.observation_space_shapes[0][0]
        # elif self.env.observation_mode == 1:
        #    self.vector_state_dimension = None
        # elif self.env.observation_mode == 2:
        #    self.vector_state_dimension = self.env.observation_space_shapes[3][0]

        self.actor = Actor(self.env.observation_space_shapes,
                           self.env.action_space.shape[0],
                           self.max_action).to(device)
        self.critic = Critic(self.env.observation_space_shapes,
                             self.env.action_space.shape[0]).to(device)

        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    def select_action(self, state):
        # if self.env.observation_mode == 0:
        #    state = torch.FloatTensor(state[0].reshape(1, -1)).to(self.device)
        state = [torch.FloatTensor(s).unsqueeze(0).to(self.device) for s in
                 state]
        return self.actor(state).cpu().data.numpy().flatten()

    @staticmethod
    def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

    def get_state_dict(self):
        state = {
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict()

        }
        return state

    def save_checkpoint(self, filename):

        torch.save(self.get_state_dict(), filename)
        """
        torch.save(self.critic.state_dict(), filename + '_critic')
        torch.save(self.critic_optimizer.state_dict(), filename + '_critic_optimizer')
        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.actor_optimizer.state_dict(), filename + '_actor_optimizer')
        """

    def save_to_onnx(self, folder='.', critic=False):
        #        device = next(model.parameters()).device
        device = self.device

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
        # print('agent train state ',state.shape)
        # print('agent train next_state',next_state.shape)
        # convert state to lit of tensors on GPU
        state = [torch.FloatTensor(s).to(self.device) for s in state]
        next_state = [torch.FloatTensor(s).to(self.device) for s in next_state]
        #        if self.env.observation_mode == 0:
        #            state = torch.FloatTensor(state[0]).to(self.device)
        #            next_state = torch.FloatTensor(next_state[0]).to(self.device)
        # print('agent train',next_state.shape)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        episode_done = torch.FloatTensor(episode_done).to(self.device)

        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + (
                (1.0 - episode_done) * self.discount * target_q).detach()
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

    def info(self):
        print("Agent Info")
        print('-----------')
        print('Not implemented yet')


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
