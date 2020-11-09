import time
#from PIL import Image
from mlagents_envs.environment import UnityEnvironment
from gym import spaces
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

engine_side_channel = EngineConfigurationChannel()
environment_side_channel = EnvironmentParametersChannel()

timerEntries = []

def main():
    unity_env = UnityEnvironment(seed = 1, side_channels =[engine_side_channel, environment_side_channel])
    #unity_env = UnityEnvironment("../envs/Berlin/Berlin_ML.exe", seed = 1, additional_args=[], side_channels =[engine_side_channel, environment_side_channel])
    engine_side_channel.set_configuration_parameters(time_scale=10, quality_level = 0)
    environment_side_channel.set_float_parameter("rewardForGoalCollision", .5)
    environment_side_channel.set_float_parameter("rewardForExplorationPointCollision", .005)
    environment_side_channel.set_float_parameter("rewardForOtherCollision", -.1)
    environment_side_channel.set_float_parameter("rewardForFallingOffMap", -1)
    environment_side_channel.set_float_parameter("rewardForEachStep", -.0001)
    environment_side_channel.set_float_parameter("segmentationMode", 1)
    #Select Vector Agent
    environment_side_channel.set_float_parameter("observationMode", 0)
    gym_env = UnityToGymWrapper(unity_env, False, False, True)#(Environment, uint8_visual, flatten_branched, allow_multiple_obs)
    observation = gym_env.reset()
    
    print(gym_env.action_space)
    print(gym_env.observation_space)
    print(observation)
    print(type(gym_env.action_space), gym_env.action_space.sample(),  type(gym_env.action_space.sample()))
 
    start_time = time.time()
    last_time = start_time
    
    BASEFILENAME = "unityGymCarStatic-walk"
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = BASEFILENAME + "-" + timestr +".csv"
    #stepTimeDeltas format: step#, steptimedelta
    stepTimeDeltas = []
    for t in range(30001):
        #action = gym_env.action_space.sample()
        # some arbitrary turns and then reverse
        if (t >= 0):
            action = [ 1, 1]
        if (t > 17000):
            action = [ 1, 0]
        if (t > 25000):
            action = [ 0, 1]
        observation, reward, done, info = gym_env.step(action)

        current_time = time.time()
        stepTimeDeltas.append("{},{}".format(t, current_time-last_time))
        last_time = current_time
        #if(t != 0 and t %10000 == 0):
            #calculateStepTime(start_time, last_time)
            #print (observation)
            #storeObservationImage(observation[0], False, t+0)
            #storeObservationImage(observation[1], True, t+1)
            #storeObservationImage(observation[2], False, t+2)
            #last_time = time.time()
        if done:
            print("\tEpisode finished after {} timesteps".format(t+1))
                
    with open(filename, "w") as logfile:
        for s in stepTimeDeltas:
            logfile.write(s+"\n") 
            
    gym_env.close()
    
def calculateStepTime(start_time, last_time):
    current_time = time.time()
    print("--- %s seconds ---" % (current_time - start_time))
    timerEntries.append(10000/(current_time-last_time))
    print("Steps/second = {}".format(timerEntries[-1]))
    averageStepTime = 0
    for j in timerEntries :
        averageStepTime += j
    print("Average Steps/second = {}".format(averageStepTime/len(timerEntries)))
 
""" 
def storeObservationImage(observation, grayscale, observationId):
    #print (observation)
    if(grayscale):
        img = Image.fromarray((observation*255).astype('uint8').reshape(observation.shape[0],observation.shape[1]), 'L')
    else:
        img = Image.fromarray((observation*255).astype('uint8'), 'RGB')
    img.save("visual_observation{}.png".format(observationId))
    #img.show()
"""
	
if __name__ == '__main__':
    main()