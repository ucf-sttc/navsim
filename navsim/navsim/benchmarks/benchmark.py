import time
import uuid
from enum import Enum
import argparse
from pathlib import Path
from PIL import Image
from mlagents_envs.environment import UnityEnvironment
from gym import spaces
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
from ..env import MapSideChannel

timerEntries = []


class ObservationMode(Enum):
    Vector = 0
    Visual = 1
    VectorVisual = 2


class RLTask(Enum):
    PointNav = 0
    SimpleObjectNav = 1
    ObjectNav = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('binary', metavar='B', help="File path to Unity game binary")
    parser.add_argument('-t', default="PointNav", choices=["PointNav", "SimpleObjectNav", "ObjectNav"],
                        help="Navigational Task that agent will train")
    parser.add_argument('-a', default="Vector", choices=["Vector", "Visual", "VectorVisual"], help="Type of agent")
    parser.add_argument('-s', default=0, type=int, choices=[0, 1, 2], help="Segmentation model for visual observations")
    parser.add_argument('-e', default=2500, type=int, help="Episode length integer")
    parser.add_argument('--worker_id', default=0, type=int,
                        help="Worker Id used to offset comms port for multiple instances")
    parser.add_argument('--seed', default=0, type=int, help="Unity Engine seed")
    parser.add_argument('--timeout', default=1000, type=int,
                        help="Amount of time in seconds to wait for a connection to the binary")
    parser.add_argument('--timescale', default=2, type=int, help="Unity Engine physics timescale modifier")
    parser.add_argument('--quality-level', default=0, type=int, help="Unity Engine quality level")
    parser.add_argument('--nophysics', action="store_true", help="Disable agent car physics simulation")
    parser.add_argument('--saveimages', action="store_true",
                        help="Save images at every visual observations at every step")
    args = parser.parse_args()

    print(args)
    # Debug
    saveImageFlag = args.saveimages
    saveOnStepIfFactorOf = 100

    # TODO bring to a config
    # Rewards
    GOAL_COLLISION_REWARD = 1
    EXP_COLLISION_REWARD = .8
    OTHER_COLLISION_REWARD = -0.9
    FALL_OFF_MAP_REWARD = -0.1
    STEP_REWARD = -0.08

    engine_side_channel = EngineConfigurationChannel()
    environment_side_channel = EnvironmentParametersChannel()
    map_side_channel = MapSideChannel()
    float_properties_side_channel = FloatPropertiesChannel()

    # Connect to Unity Editor environment
    unityEnvironmentStr = str(args.binary) if args.binary else None
    # Connect to specified binary environment
    unity_env = UnityEnvironment(file_name=unityEnvironmentStr, worker_id=args.worker_id, seed=args.seed,
                                 timeout_wait=args.timeout,
                                 side_channels=[engine_side_channel, environment_side_channel, map_side_channel, float_properties_side_channel])

    # Engine Side Channels
    engine_side_channel.set_configuration_parameters(time_scale=args.timescale, quality_level=args.quality_level)

    # Rewards
    environment_side_channel.set_float_parameter("rewardForGoalCollision", GOAL_COLLISION_REWARD)
    environment_side_channel.set_float_parameter("rewardForExplorationPointCollision", EXP_COLLISION_REWARD)
    environment_side_channel.set_float_parameter("rewardForOtherCollision", OTHER_COLLISION_REWARD)
    environment_side_channel.set_float_parameter("rewardForFallingOffMap", FALL_OFF_MAP_REWARD)
    environment_side_channel.set_float_parameter("rewardForEachStep", STEP_REWARD)

    # Segmentation
    # Object Segmentation 0
    # Tag Segmentation 1
    # Layer Segmentation 2
    segmentationMode = args.s
    environment_side_channel.set_float_parameter("segmentationMode", segmentationMode)

    # Agent Selection
    observationModeIndex = ObservationMode[args.a].value
    environment_side_channel.set_float_parameter("observationMode", observationModeIndex)

    # Agent Car Physics
    # if flag is set, hence true, then set agentCarPhysics to 0, false then 1
    agentCarPhysics = int(not args.nophysics)
    environment_side_channel.set_float_parameter("agentCarPhysics", agentCarPhysics)

    # Episode Length
    episodeLength = args.e
    environment_side_channel.set_float_parameter("episodeLength", episodeLength)

    # Task Selection
    selectedTaskIndex = RLTask[args.t].value
    environment_side_channel.set_float_parameter("selectedTaskIndex", selectedTaskIndex)

    # Goal Selection
    # 0 - Tocus;
    # 1 - sedan1
    # 2 - Car1
    # 3 - Car2
    # 4 - City Bus
    # 5 - Sporty_Hatchback
    # Else - SEDAN
    # goalSelectionIndex = 0
    # environment_side_channel.set_float_parameter("goalSelectionIndex", goalSelectionIndex)

    # File to save step time benchmarks
    BASEFILENAME = "bench" + uuid.uuid4().hex + "-a" + args.a + "-t" + str(args.t) + "-s" + str(
        segmentationMode) + "-e" + str(episodeLength) + "-p" + str(agentCarPhysics) + "-sv" + str(int(saveImageFlag))
    print(BASEFILENAME)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    baseFileNameWithTime = BASEFILENAME + "-" + timestr
    csvFilename = baseFileNameWithTime + ".csv"
    # Visual Observation Save outs
    if (saveImageFlag and (observationModeIndex == 1 or observationModeIndex == 2)):
        imgBasePath = Path("") / Path(baseFileNameWithTime)
        imgBasePath.mkdir(parents=True, exist_ok=True)

    # Create Gym Environment
    gym_env = UnityToGymWrapper(unity_env, False, False, True)
    observation = gym_env.reset()

    print('=================================')
    print(gym_env.action_space)
    print(gym_env.action_space.sample())
    print(type(gym_env.action_space), gym_env.action_space.sample(), type(gym_env.action_space.sample()))
    print(gym_env.observation_space)
    print(observation)
    print(gym_env.metadata)
    print(gym_env.spec)
    print(gym_env.name)
    print(gym_env._env.get_steps(gym_env.name))
    print(gym_env.action_space.high, '+++++', gym_env.action_space.low)
    print('=================================')

    start_time = time.time()
    last_time = start_time

    # stepTimeDeltas format: step#, steptimedelta
    stepTimeDeltas = []
    for t in range(30001):
        # action = gym_env.action_space.sample()
        # some arbitrary turns and then reverse
        if (t >= 0):
            action = [1, 1, 0]
        if (t > 17000):
            action = [1, 0, 0]
        if (t > 25000):
            action = [0, 1, 0]
        observation, reward, done, info = gym_env.step(action)

        current_time = time.time()
        stepTimeDeltas.append("{},{}".format(t, current_time - last_time))
        last_time = current_time
        if (saveImageFlag and t % saveOnStepIfFactorOf == 0 and (
                observationModeIndex == 1 or observationModeIndex == 2)):
            # calculateStepTime(start_time, last_time)
            # print (observation)
            storeObservationImage(imgBasePath, observation[0], False, t + 0)
            storeObservationImage(imgBasePath, observation[1], True, t + 1)
            storeObservationImage(imgBasePath, observation[2], False, t + 2)
            # last_time = time.time()
        if done:
            print("\tEpisode finished after {} timesteps".format(t + 1))

    with open(csvFilename, "w") as logfile:
        for s in stepTimeDeltas:
            logfile.write(s + "\n")

    gym_env.close()


def calculateStepTime(start_time, last_time):
    current_time = time.time()
    print("--- %s seconds ---" % (current_time - start_time))
    timerEntries.append(10000 / (current_time - last_time))
    print("Steps/second = {}".format(timerEntries[-1]))
    averageStepTime = 0
    for j in timerEntries:
        averageStepTime += j
    print("Average Steps/second = {}".format(averageStepTime / len(timerEntries)))


def storeObservationImage(basepath, observation, grayscale, observationId):
    # print (observation)
    if (grayscale):
        img = Image.fromarray((observation * 255).astype('uint8').reshape(observation.shape[0], observation.shape[1]),
                              'L')
    else:
        img = Image.fromarray((observation * 255).astype('uint8'), 'RGB')
    img_path = basepath / f"visual_observation{observationId}.png"
    img.save(img_path)
    # img.show()

# For python debugger to directly run this script
if __name__ == "__main__":
    main()
