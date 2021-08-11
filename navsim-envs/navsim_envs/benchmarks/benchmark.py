import gym
import time
import sys
from pathlib import Path
from copy import deepcopy

import navsim_envs
from navsim_envs.env import AroraGymEnv
from navsim_envs.util.configs import env_config as env_conf
from navsim_envs.util.configs import config_banner, load_config

def create_run_folder():
    run_base_folder = Path('.').resolve() / 'benchmark_logs'
    run_base_folder.mkdir(parents=True, exist_ok=True)
    return run_base_folder 

def env_config():
    config = deepcopy(env_conf)
    config["log_folder"] = str(create_run_folder() / "env_log")
    if len(sys.argv) > 1:
        env_config_file = sys.argv[1]
        env_config_from_file = load_config(env_config_file)
        if env_config_from_file is not None:
            config.update(env_config_from_file)
    return config

def calculateStepTime(start_time, last_time):
    current_time = time.time()
    print("--- %s seconds ---" % (current_time - start_time))
    timerEntry = (10000 / (current_time - last_time))
    print("Steps/second = {}".format(timerEntries[-1]))

def averageTime(timerEntries):
    averageStepTime = 0
    for j in timerEntries:
        averageStepTime += j[1]
    avgtime=averageStepTime / len(timerEntries)
    print("Average Steps/second = {}".format(avgtime))
    return avgtime

def main():
        print("Starting Navsim Benchmark")
        timestr = time.strftime("%Y%m%d-%H%M%S")
        csvFilename = "bench-" + timestr + ".csv"
        run_folder = create_run_folder()
        csv_loc = run_folder / csvFilename

        env = gym.make("arora-v0", env_config=env_config())
        throttle = 1
        steering = 0

        start_time = time.time()
        last_time = start_time
        stepTimeDeltas = []
        o = env.reset()
        for t in range(30001):
            o, r, done, i = env.step([throttle, steering, -1])
            current_time = time.time()
            stepDelta = current_time - last_time
            stepTimeDeltas.append([t, stepDelta])
            last_time = current_time
            if done:
                print("\tEpisode finished after {} timesteps".format(t + 1))
                o = env.reset()
        averageTime(stepTimeDeltas)
        with open(csv_loc, "w") as logfile:
            logfile.write("s, stepDelta")
            for s in stepTimeDeltas:
                logfile.write(f'{s[0]}, {s[1]}')
