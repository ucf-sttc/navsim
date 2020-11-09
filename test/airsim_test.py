import unittest

from pathlib import Path
import sys
import csv
from tqdm import tqdm

#===== TODO: Remove this when ezai_util is an installed package
base_path=Path('..')

for pkg in ['ezai_util']:
    pkg_path = base_path / pkg
    pkg_path = str(pkg_path.resolve())
    print(pkg_path)
    if not pkg_path in sys.path:
        sys.path.append(pkg_path)
#===== TODO: Remove this when ezai_util is an installed package

import ezai_util
import airsim

from ezai_util import DictObj, ResourceCounter

conf = DictObj().load_from_json_file('airsim_test_conf.json')
rl_conf = DictObj(conf.rl_conf)
env_conf = DictObj(conf.env_conf)

t_max = env_conf.max_steps

def helper(env_conf):
    testlog_filename = f'airsim_test_obs_mode_{env_conf.observation_mode}.log'
    with open(testlog_filename, mode='w+') as testlog_file:
        log_str = f'testing with observation_mode {env_conf.observation_mode}'
        testlog_file.write(log_str)
        print(log_str)
        testlog_file.flush()

        resources_filename = f'airsim_test_obs_mode_{env_conf.observation_mode}.csv'
        rc = ResourceCounter()
        rc.start()

        env = airsim.AirSimEnv(env_conf)
        time_since_start, current_memory, peak_memory = rc.stop()
        log_str = f'Unity env creation resource usage: ' \
                  f'time:{time_since_start},peak_memory:{peak_memory},current_memory:{current_memory}'
        testlog_file.write(log_str)
        print(log_str)
        testlog_file.flush()

        try:
            env.info()

            # create a memory for random play
            #memory = airsim.Memory(capacity = rl_conf.memory_capacity,
            #                       state_shapes = env.observation_space_shapes,
            #                       action_shape = env.action_space_shape,
            #                       seed = env.seed)
            #memory.info()

            # do the random play and fill the memory
            resources_file = open(resources_filename, mode='w+')
            writer = csv.writer(resources_file,
                                delimiter=',',
                                quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                ['episode_num','t','r','step_time', 'unity_step_time', 'peak_memory'])
            resources_file.flush()


            for episode_num in tqdm(range(0,int(rl_conf['num_episodes']))):

                t=0
                episode_done = False
                last_snapshot_time = rc.start()

                #1. observe initial state
                s = env.reset()

                while not episode_done:

#                    time_since_start, current_memory, peak_memory = rc.snapshot()
                    step_resources=[[rc.snapshot()]] #0
                    t += 1

                    #2. select an action, and observe the next state
                    a = env.action_space.sample()

                    step_resources.append([rc.snapshot()]) #1
                    s_, r, episode_done, info = env.step(a)
                    step_resources.append([rc.snapshot()]) #2

                    #if(t == t_max):
                    #    episode_done=True

                    #3. save in memory

                    #s = np.asarray(s).squeeze()
                    #a = np.asarray(a).squeeze()
                    #r = np.asarray(r).squeeze()
                    #s_ = None if episode_done else s_ #np.asarray(s_).squeeze()
                    #memory.append(s=s,a=a,r=r,s_=s_, d=episode_done)

                    if (t_max and t == t_max):
                        break

                # sample the memory
                #s,a,r,s_,d = memory.sample(100)
                #airsim.Memory.sample_info(s,a,r,s_,d)
                    step_resources.append([rc.snapshot()])  #3

                    step_time = step_resources[3][0]-step_resources[0][0]
                    #step_peak_memory = max(step_resources[3][2],step_resources[0][2])
                    unity_step_time = step_resources[2][0]-step_resources[1][0]
                    #unity_step_peak_memory = max(step_resources[3][2],step_resources[0][2])
                    peak_memory = step_resources[3][2]

                    writer.writerow(
                        [episode_num,t,r,step_time, unity_step_time, peak_memory])
                    resources_file.flush()

                    # Time Step End Processing
                    s = s_

        finally:
            resources_file.close()
            env.close()
            # save results


class TestAirSim(unittest.TestCase):

    def test_observation_mode_0(self):
        env_conf1 = env_conf.deepcopy()
        env_conf1.observation_mode = 0
        helper(env_conf1)
        #self.assertTrue(tf.test.is_built_with_cuda())


if __name__ == '__main__':
    unittest.main()