import time
import argparse
from pathlib import Path
from PIL import Image
from mlagents_envs.environment import UnityEnvironment
from gym import spaces
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
import socket

def main(args : argparse.Namespace):

    episodes = 50
    for unitySeed in [1,123,999]:
        for episode_sequence in range(10):
            #Rewards
            GOAL_COLLISION_REWARD = 1
            EXP_COLLISION_REWARD = .8
            OTHER_COLLISION_REWARD = -0.9
            FALL_OFF_MAP_REWARD = -0.1
            STEP_REWARD = -0.08

            MAX_STEPS=5

            engine_side_channel = EngineConfigurationChannel()
            environment_side_channel = EnvironmentParametersChannel()

            #Connect to Unity Editor environment
            unityEnvironmentStr = None
            #Connect to specified binary environment
            #unityEnvironmentStr = "../envs/Berlin_v2/Berlin_Walk_v2.exe"
            unityEnvironmentStr = "/data/work/unity-envs/Build2.6.3/Berlin_Walk_V2"
            unity_env = UnityEnvironment(file_name = unityEnvironmentStr, seed = unitySeed, timeout_wait=1000, side_channels =[engine_side_channel, environment_side_channel])

            #Engine Side Channels
            engine_side_channel.set_configuration_parameters(time_scale=1, quality_level = 0)

            #Rewards
            environment_side_channel.set_float_parameter("rewardForGoalCollision", GOAL_COLLISION_REWARD)
            environment_side_channel.set_float_parameter("rewardForExplorationPointCollision", EXP_COLLISION_REWARD)
            environment_side_channel.set_float_parameter("rewardForOtherCollision", OTHER_COLLISION_REWARD)
            environment_side_channel.set_float_parameter("rewardForFallingOffMap", FALL_OFF_MAP_REWARD)
            environment_side_channel.set_float_parameter("rewardForEachStep", STEP_REWARD)

            environment_side_channel.set_float_parameter("observationMode",0) # vector agent
            environment_side_channel.set_float_parameter("episodeLength", MAX_STEPS+2)
            environment_side_channel.set_float_parameter("selectedTaskIndex", 0) # pointnav

            #File to save step time benchmarks
            BASEFILENAME = f"{socket.gethostname()}-seed-{unitySeed}-seq-{episode_sequence}"
            print(BASEFILENAME)
            timestr = time.strftime("%Y%m%d-%H%M%S")
            baseFileNameWithTime = BASEFILENAME + "-" + timestr
            csvFilename = BASEFILENAME + ".csv"

            #Create Gym Environment
            gym_env = UnityToGymWrapper(unity_env, False, False, True)

            with open(csvFilename, "w") as logfile:
                for t in range(episodes):
                    print("Starting episode",t)
                    s = gym_env.reset()
                    for i in range(MAX_STEPS):
                        o = gym_env.step(gym_env.action_space.sample())
                    logfile.write(','.join(str(e) for e in s[0])+'\n')
            gym_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)