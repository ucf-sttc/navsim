from collections import OrderedDict

import math
import torch
from torch.nn import functional as F

class ModelTorch(torch.nn.Module):
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


class ActorCriticWrapper(torch.nn.Module):
    """Wraps Actor and Critic for the purpose of TensorBoard

    """
    def __init__(self, state_dimensions, action_dimension, max_action):
        super().__init__()
        self.actor = Actor(state_dimensions, action_dimension, max_action)
        self.critic = Critic(state_dimensions, action_dimension)

    def forward(self, state, action):
        q1 = self.actor(state)
        q2 = self.critic(state, action)
        return q1, q2


class Actor(torch.nn.Module):
    """Actor class
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

        # input layer
        for state_dim in state_dimensions:
            if len(state_dim) == 1:  # means we have vector observation
                out_size = state_dim[0]*16  # make sure its always int
                l_out_size.append(out_size)
                layer = torch.nn.Sequential(OrderedDict([
                    (f'linear_1',torch.nn.Linear(state_dim[0], state_dim[0]*4)),
                    (f'activ_1',torch.nn.ReLU()),
                    (f'linear_2',torch.nn.Linear(state_dim[0]*4, state_dim[0]*8)),
                    (f'activ_2',torch.nn.ReLU()),
                    (f'linear_3',torch.nn.Linear(state_dim[0]*8, state_dim[0]*16)),
                    (f'activ_3',torch.nn.ReLU()),
                    #(f'linear_4',torch.nn.Linear(state_dim[0]*16, state_dim[0]*8)),
                    #(f'activ_4',torch.nn.ReLU()),
                    #(f'linear_5',torch.nn.Linear(state_dim[0]*8, state_dim[0]*4)),
                    #(f'activ_5',torch.nn.ReLU()),
                    #(f'linear_6',torch.nn.Linear(state_dim[0]*4, state_dim[0])),
                    #(f'activ_6',torch.nn.ReLU()),
                ]))
                self.feature_layers.append(layer)

            else:  # visual:
                layer = torch.nn.Sequential(OrderedDict([
                    ('conv_1',torch.nn.Conv2d(in_channels=state_dim[2],
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation)),
                    # [batch_size, n_features_conv, height, width]
                    ('pool_1',torch.nn.MaxPool2d(kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation)),
                    ('activ_1',torch.nn.ReLU())
                ]))
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

        self.out_layer = torch.nn.Sequential(OrderedDict([
            ('linear_out',torch.nn.Linear(cat_dim, action_dimension)),
            ('activ_out',torch.nn.Tanh())
        ]))
        # print(f'{l_out_size},cat_dim:{cat_dim},action_dim:{action_dimension}')
        # self.action_out = torch.nn.Linear(cat_dim, action_dimension)

        self.max_action = max_action

    def forward(self, state):
        """

        :param state: state is a list of numpy/cupy arrays
        :return:
        """
        # curframe = inspect.currentframe()
        # calframe = inspect.getouterframes(curframe, 2)
        # print('actor caller name:', calframe[1][3])
        # print('actor',state.shape)
        # print('actor',self.l1)
        # state = state[0]
        #state = [torch.as_tensor(s,
        #                         dtype=torch.float,
        #                         device=self.device).unsqueeze(0) for s in
        #         state]
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
    """Critic class
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
                out_size = state_dim[0]*16  # make sure its always int
                l_out_size.append(out_size)
                layer = torch.nn.Sequential(OrderedDict([
                    (f'linear_1',torch.nn.Linear(state_dim[0], state_dim[0]*4)),
                    (f'activ_1',torch.nn.ReLU()),
                    (f'linear_2',torch.nn.Linear(state_dim[0]*4, state_dim[0]*8)),
                    (f'activ_2',torch.nn.ReLU()),
                    (f'linear_3',torch.nn.Linear(state_dim[0]*8, state_dim[0]*16)),
                    (f'activ_3',torch.nn.ReLU()),
                    #(f'linear_4',torch.nn.Linear(state_dim[0]*16, state_dim[0]*8)),
                    #(f'activ_4',torch.nn.ReLU()),
                    #(f'linear_5',torch.nn.Linear(state_dim[0]*8, state_dim[0]*4)),
                    #(f'activ_5',torch.nn.ReLU()),
                    #(f'linear_6',torch.nn.Linear(state_dim[0]*4, state_dim[0])),
                    #(f'activ_6',torch.nn.ReLU()),
                ]))
                self.feature_layers.append(layer)
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
        #state = [torch.as_tensor(s,
        #                         dtype=torch.float,
        #                         device=self.device).unsqueeze(0) for s in
        #         state]

        #action = torch.as_tensor(action,
        #                         dtype=torch.float,
        #                         device=self.device)

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


def num2tuple(num):
    return tuple(num) if isinstance(num, (tuple, list)) else (num, num)