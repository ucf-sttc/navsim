import math
import argparse
import random
from collections import deque
import datetime
import time

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

move_speed=50
rotational_speed=5

class SimpleAgent(object):
    def __init__(self, action_space, lr=0.01, batch_size=64, discount_rate=1.0, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01):
        self.action_space=action_space
        self.memory=deque(maxlen=1000)
        self.lr=lr
        self.batch_size=batch_size
        self.discount_rate=discount_rate
        self.exploration_rate=exploration_rate
        self.exploration_decay=exploration_decay
        self.exploration_min=exploration_min

        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size=2, input_shape=(84, 84, 3), activation='tanh'))
        self.model.add(Flatten())
        self.model.add(Dense(4, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.lr))

    def predict (self, observation):
        return self.model.predict(observation)
        
    def act(self, observation, ex_rate):
        return np.random.randint(0,4) if np.random.rand() <= ex_rate else  np.argmax(self.model.predict(observation)[0])

    def calc_exploration_rate(self, t):
        return max(self.exploration_min, min(self.exploration_rate, 1.0 - math.log10((t + 1) * self.exploration_decay)))

    def memorize(self, observation, action, reward, next_observation, done):
        self.memory.append((observation, action, reward, next_observation, done))

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        for observation, action, reward, next_observation, done in minibatch:
            #print("*****", observation, action, reward, next_observation, done, "*****")
            y_target = self.model.predict(observation)
            #print(action, y_target, y_target[0][action], np.max(y_target[0]))
            y_target[0][action] = reward if done else reward + self.discount_rate * np.max(self.model.predict(next_observation)[0])
            #print(observation, y_target)
            x_batch.append(observation[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=1)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay 

    def save(self, filename):
        self.model.save(filename)

def main(args : argparse.Namespace):
    print(args, type(args))

    #env = gym.make(args.env_id)
    #env.seed(0)
    engine_side_channel = EngineConfigurationChannel()
    environment_side_channel = EnvironmentParametersChannel()
    unity_env = UnityEnvironment(seed = np.random.randint(0,100000), side_channels =[engine_side_channel, environment_side_channel])
    engine_side_channel.set_configuration_parameters(time_scale=10, quality_level = 0)
    environment_side_channel.set_float_parameter("rewardForGoalCollision", 1)
    environment_side_channel.set_float_parameter("rewardForExplorationPointCollision", .8)
    environment_side_channel.set_float_parameter("rewardForOtherCollision", .3)
    environment_side_channel.set_float_parameter("rewardForFallingOffMap", .1)
    environment_side_channel.set_float_parameter("rewardForEachStep", .0001)
    environment_side_channel.set_float_parameter("segmentationMode", 0)
    environment_side_channel.set_float_parameter("observationMode", 1)
    gym_env = UnityToGymWrapper(unity_env, False, False, True)#(Environment, uint8_visual, flatten_branched, allow_multiple_obs)
    
    print("*****", gym_env.__dict__, "*****")
    print(type(gym_env.__dict__['name']))
    print(gym_env.__dict__['_env'], gym_env.__dict__['name'], "Action Space:", gym_env.action_space, "Observation Space:", gym_env.observation_space, "Reward Range:", gym_env.reward_range)
    print("Observation Space Ranges")
    for ob in gym_env.observation_space:
        print(ob.high, '+++++', ob.low)
        print(type(ob.high), '+++++', type(ob.low))
        print(ob.high.shape, '+++++', ob.low.shape)
        print(type(ob))
        print('========')
    print("Action Space Sample", gym_env.action_space.sample())
    agent = SimpleAgent(gym_env.action_space, lr=0.01)


    episode_count = 10
    scores = deque(maxlen=100)

    start_time = time.time()
    last_time = start_time
    cur_step = 0
    step_average_interval=250
    
    
    BASEFILENAME = "DQN_visual_car_agent-walk"
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = BASEFILENAME + "-" + timestr +".csv"
    #stepTimeDeltas format: step#, steptimedelta, episode_count, agent_act_time, agent_memorize_time
    stepTimeDeltas = []
    for i in range(episode_count):
        done = False
        ob = gym_env.reset()[0]
        score = 0
        while not done:
            #action = [rotated left, rotate right, forward, backward]
            ob = np.reshape(ob, [1,*ob.shape])
            a = datetime.datetime.now()
            action = agent.act(ob, agent.calc_exploration_rate(i))
            b = datetime.datetime.now()
            agent_act_time = b-a
            
            action=0
            
            true_action = []
            if action==0:
                true_action = [move_speed,rotational_speed]
            elif action==1:
                true_action = [move_speed,rotational_speed]
                
            #print("Episode:", i, "Observation", ob, "Agent Action", action)
            next_ob, reward, done, _ = gym_env.step(true_action)
            next_ob = next_ob[0]
            next_ob = np.reshape(next_ob, [1,*next_ob.shape])
            
            a = datetime.datetime.now()
            agent.memorize(ob, action, reward, next_ob, done)
            b = datetime.datetime.now()
            agent_memorize_time = b-a
            
            ob = next_ob 
            score += reward
            
            current_time = time.time()
            stepTimeDeltas.append("{},{},{},{},{},{}".format(cur_step, current_time-last_time, i, agent_act_time, agent_memorize_time, ""))
            last_time = time.time()
            
            cur_step += 1
            
        scores.append(score)
        mean_score = np.mean(scores)

        #if i % 10 == 0:
        #    print('[Episode {}] - Mean survival score over last 100 episodes was {}'.format(i, mean_score))
        
        a = datetime.datetime.now()
        agent.replay(agent.batch_size)
        b = datetime.datetime.now()
        agent_replay_time = b-a
        
        current_time = time.time()
        stepTimeDeltas.append("{},{},{},{},{},{}".format("", current_time-last_time, i, "", "", agent_replay_time))
        last_time = time.time()

    with open(filename, "w") as logfile:
        for s in stepTimeDeltas:
            logfile.write(s+"\n")
            
    agent.save("Model")

    gym_env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('null', nargs='?', default='null', help='Nothing for now')
    args = parser.parse_args()
    main(args)
