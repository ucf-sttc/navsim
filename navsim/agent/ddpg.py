import torch
import torch.nn.functional as F

from copy import deepcopy

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


from navsim.agent.nn import ActorCriticWrapper, Actor, Critic


class DDPGAgent(object):

    NN_WRAPPER = ActorCriticWrapper

    def __init__(self, env, device, discount=0.99, tau=0.005):
        self.env = env
        self.device = device
        self.discount = discount
        self.tau = tau
        # self.state_dim = state_dim

        self.max_action = self.env.action_space.high[0]

        # if self.env.obs_mode == 0:
        #    self.vector_state_dimension = self.env.observation_space_shapes[0][0]
        # elif self.env.obs_mode == 1:
        #    self.vector_state_dimension = None
        # elif self.env.obs_mode == 2:
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

        self.actor_loss = None
        self.critic_loss = None

    def select_action(self, state):
        # if self.env.obs_mode == 0:
        #    state = torch.FloatTensor(state[0].reshape(1, -1)).to(self.device)
        state = [torch.as_tensor(s,
                                 dtype=torch.float,
                                 device=self.device).unsqueeze(0) for s in
                 state]
        return self.actor(state).data.cpu().numpy().flatten()

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
        # convert state to list of tensors on GPU
        state = [torch.as_tensor(s,
                                 dtype=torch.float,
                                 device=self.device) for s in state]
        next_state = [torch.as_tensor(s,
                                      dtype=torch.float,
                                      device=self.device) for s in next_state]
        action = torch.as_tensor(action, dtype=torch.float, device=self.device)
        target_q = self.critic_target(next_state, self.actor_target(next_state))

        reward = torch.as_tensor(reward, dtype=torch.float, device=self.device)
        episode_done = torch.as_tensor(episode_done, dtype=torch.float,
                                   device=self.device)
        #reward = target_q.new(reward)
        #episode_done = target_q.new(episode_done)
        target_q = reward + (
                (1.0 - episode_done) * self.discount * target_q).detach()

        current_q = self.critic(state, action)
        self.critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()
        self.actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        self.actor_loss.backward()
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
