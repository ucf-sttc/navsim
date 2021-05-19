import csv
import cv2
from pathlib import Path


import torch
from tqdm import tqdm
import numpy as np
import math

from navsim import NavSimGymEnv, DDPGAgent, Memory
from navsim.util import sizeof_fmt, image_layout
from navsim.util import ObjDict, ResourceCounter

import traceback
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def s_hwc_to_chw(s):
    # TODO: HWC to CHW conversion optimized here
    # because pytorch can only deal with images in CHW format
    # we are making the optimization here to convert from HWC to CHW format.
    if isinstance(s, np.ndarray) and (s.ndim > 2):  # state is ndarray
        s = image_layout(s, 'hwc', 'chw')
    elif isinstance(s, list):  # state is a list of states
        for i in range(len(s)):
            if isinstance(s[i], np.ndarray) and (s[i].ndim > 2):  # state is ndarray
                s[i] = image_layout(s[i], 'hwc', 'chw')
    return s


class Executor:
    """
        TODO
    """

    def __init__(self, run_id='navsim_demo',
                 resume=True, conf=None):
        """

        :param conf: A ObjDict containing dictionary and object interface
        :param env: Any gym compatible environment
        :param resume: True: means continue if run exists, else start new
                           False: means overwrite if exists, else start new
        """
        self.run_id = run_id

        self.run_base_folder = Path(self.run_id)
        self.run_base_folder_str = str(self.run_base_folder.resolve())
        #        if run_base_folder.is_dir():
        #            raise ValueError(f"{run_base_folder_str} exists as a non-directory. "
        #                             f"Please remove the file or use a different run_id")
        if resume and self.run_base_folder.is_dir():
            self.conf = ObjDict().load_from_json_file(f"{self.run_base_folder_str}/conf.json")
            self.resume = True
            self.file_mode = 'a+'

        # else just start fresh
        else:
            self.conf = conf
            self.resume = False
            self.run_base_folder.mkdir(parents=True, exist_ok=True)
            self.conf.save_to_json_file(f"{self.run_base_folder_str}/conf.json")
            self.file_mode = 'w+'

        self.run_conf = ObjDict(self.conf.run_config)
        self.env_config = ObjDict(self.conf.env_config)

        pylog_filename = self.run_base_folder / 'py.log'  # TODO: use logger
        self.pylog_filename = str(pylog_filename.resolve())
        step_results_filename = self.run_base_folder / 'step_results.csv'
        self.step_results_filename = str(step_results_filename.resolve())
        episode_results_filename = self.run_base_folder / 'episode_results.csv'
        self.episode_results_filename = str(episode_results_filename.resolve())
        env_log_folder = self.run_base_folder / 'env.log'
        self.env_config.log_folder = str(env_log_folder.resolve())
        self.model_filename = f"{self.run_base_folder_str}/model_state.pt"
        self.memory_filename = f"{self.run_base_folder_str}/memory.pkl"
        # TODO: Add the code to delete previous files
        # TODO: Add the code to add categories
        self.summary_writer = SummaryWriter(f"{self.run_base_folder_str}/tb")

        self.rc = ResourceCounter()
        self.files_open()

        try:
            self.env = None
            self.env_open()
            if resume and self.run_base_folder.is_dir():
                self.memory = Memory.load_from_pkl(self.memory_filename)
            else:
                # TODO: HWC to CHW conversion optimized here
                # because pytorch can only deal with images in CHW format
                # we are making the optimization here to convert from HWC to CHW format.
                memory_observation_space_shapes = []
                for item in self.env.observation_space_shapes:
                    if len(item) == 3:  # means its an image in HWC
                        memory_observation_space_shapes.append((item[2], item[1], item[0]))
                    else:
                        memory_observation_space_shapes.append(item)
                self.memory = Memory(
                    capacity=self.run_conf.memory_capacity,
                    state_shapes=memory_observation_space_shapes,
                    action_shape=self.env.action_space.shape,
                    seed=self.run_conf['seed']
                )
                self.memory.info()

            self.max_action = self.env.action_space.high[0]
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            self.agent = DDPGAgent(
                env=self.env,
                device=self.device,
                discount=self.run_conf['discount'], tau=self.run_conf['tau']
            )

            # ReplayBuffer(self.vector_state_dimension, self.action_dimension)

            self.apply_seed()

        # TODO: Enable tensorboard logging
        #        if self.enable_logging:
        #            from torch.utils.tensorboard import SummaryWriter
        #            self.writer = SummaryWriter('./logs/' + self.run_conf['env_name'] + '/')
        except Exception as e:
            self.env_close()
            self.files_close()
            print(traceback.format_exc())
            # print(e)

    # TODO: Find a better name for  this function
    def write_tb_values(self, values, t: int):
        for key, value in values.items():
            self.summary_writer.add_scalar(key, value, t)
            self.summary_writer.flush()

    def apply_seed(self):
        self.env.seed(self.run_conf['seed'])  # TODO: not needed because env is seeded at time of creation
        torch.manual_seed(self.run_conf['seed'])
        np.random.seed(self.run_conf['seed'])

    def files_open(self):
        self.pylog_file = open(self.pylog_filename, mode=self.file_mode)

        self.step_results_file = open(self.step_results_filename, mode=self.file_mode)
        self.step_results_writer = csv.writer(self.step_results_file,
                                              delimiter=',',
                                              quotechar='"',
                                              quoting=csv.QUOTE_MINIMAL)
        if not self.resume:
            self.step_results_writer.writerow(
                ['episode_num',
                 't',
                 'r',
                 'step_time',
                 'unity_step_time',
                 'peak_memory'])
        self.step_results_file.flush()

        self.episode_results_file = open(self.episode_results_filename, mode=self.file_mode)
        self.episode_results_writer = csv.writer(self.episode_results_file,
                                                 delimiter=',',
                                                 quotechar='"',
                                                 quoting=csv.QUOTE_MINIMAL)
        if not self.resume:
            self.episode_results_writer.writerow(
                ['episode_num',
                 'episode_reward',
                 'episode_time',
                 'episode_peak_memory'])
        self.episode_results_file.flush()

    def files_close(self):
        self.pylog_file.close()
        self.step_results_file.close()
        self.episode_results_file.close()

    def env_open(self):
        self.rc.start()
        self.env = NavSimGymEnv(self.env_config)
        self.env.reset()
        time_since_start, current_memory, peak_memory = self.rc.stop()
        log_str = f'Unity env creation resource usage: \n' \
                  f'time:{time_since_start},peak_memory:{sizeof_fmt(peak_memory)},' \
                  f'current_memory:{sizeof_fmt(current_memory)}\n'
        self.pylog_file.write(log_str)
        print(log_str)
        self.pylog_file.flush()
        self.env.info()
        #self.env.info_steps(save_visuals=True)

    def env_close(self):
        if self.env is None:
            print("Env is None")
        else:
            print("closing the env now")
            self.env.close()

    def execute(self):
        """
        Execute for the number of episodes
        :return:
        """
        t_max = int(self.run_conf['episode_max_steps'])
        num_episodes = int(self.run_conf['num_episodes'])
        checkpoint_interval = int(self.run_conf['checkpoint_interval'])
        num_episode_blocks = int(math.ceil(num_episodes / checkpoint_interval))

        for i in range(0,num_episode_blocks):
            for episode_num in tqdm(iterable=range((i*checkpoint_interval)+1, min((i+1)*checkpoint_interval,num_episodes)+1),
                                    desc=f"Episode {(i*checkpoint_interval)+1}-{min((i+1)*checkpoint_interval,num_episodes)}/{num_episodes}",
                                    unit='episode', ascii=True, ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]'):
                self.rc.start()
                episode_resources = [self.rc.snapshot()]  # e0

                # initialise the episode counters
                episode_done = False
                episode_reward = 0
                t = 0

                self.env.start_navigable_map()
                # observe initial s
                s = self.env.reset()  # s comes out of env as a tuple always
                #self.env.step(self.env.action_space.sample())
                # get the navigable map and save it as image
                navigable_map = self.env.get_navigable_map()
                if navigable_map is not None:
                    cv2.imwrite(str((self.run_base_folder / f'navigable_map_{episode_num}.jpg').resolve()),navigable_map*255)
                else:
                    print(f'Map for episode {episode_num} is None')
                # TODO: HWC to CHW conversion optimized here
                # because pytorch can only deal with images in CHW format
                # we are making the optimization here to convert from HWC to CHW format.
                s = s_hwc_to_chw(s)

                samples_before_training = (self.run_conf['batch_size'] * self.run_conf['batches_before_train'])
                while not episode_done:
                    step_resources = [self.rc.snapshot()]  # s0
                    t += 1

                    # do the random sampling until enough memory is full
                    if self.memory.size < samples_before_training:
                        a = self.env.action_space.sample()
                    else:
                        # TODO: Find the best place to train, moved here for now
                        batch_s, batch_a, batch_r, batch_s_, batch_d = self.memory.sample(self.run_conf['batch_size'])
                        # print('training the agent')
                        self.agent.train(batch_s, batch_a, batch_r, batch_s_, batch_d)

                        a = (self.agent.select_action(s) + np.random.normal(
                            0, self.max_action * self.run_conf['expl_noise'],
                            size=self.env.action_space_shape[0])
                             ).clip(
                            -self.max_action,
                            self.max_action
                        )

                    step_resources.append(self.rc.snapshot())  # s1
                    s_, r, episode_done, info = self.env.step(a)

                    # TODO: HWC to CHW conversion optimized here
                    # because pytorch can only deal with images in CHW format
                    # we are making the optimization here to convert from HWC to CHW format.
                    s_ = s_hwc_to_chw(s_)

                    #    s = [[s]]  # make state a list of sequences
                    #    s_ = [[s_]]
                    # elif isinstance(s[0], (int,float)):  # state does not have multiple seq
                    # s = [s]
                    # s_ = [s_]
                    # for item in s_:

                    step_resources.append(self.rc.snapshot())  # s2

                    self.memory.append(
                        s=s, a=a, s_=s_, r=r,
                        d=float(episode_done))  # if t < t_max -1 else 1)
                    s = s_

                    episode_reward += r

                    # if self.memory.size >= self.run_conf['batch_size'] * self.run_conf['batches_before_train']:

                    #                    if (t >= self.config['batch_size'] * self.config['batches_before_train']) and (t % 1000 == 0):
                    # episode_evaluations.append(evaluate_policy(self.agent, self.env, self.config['seed']))

                    step_resources.append(self.rc.snapshot())  # 3

                    step_time = step_resources[3][0] - step_resources[0][0]
                    unity_step_time = step_resources[2][0] - step_resources[1][0]
                    peak_memory = step_resources[3][2]

                    # TODO: Collect these through tensorboard
                    self.step_results_writer.writerow(
                        [episode_num, t, r, step_time, unity_step_time, peak_memory])
                    self.step_results_file.flush()
                    if (t_max and t >= t_max):
                        break
                    # t += 1
                # end of while loop
                # episode end processing
                episode_resources.append(self.rc.stop())  # e1
                episode_time = episode_resources[1][0] - episode_resources[0][0]
                episode_peak_memory = episode_resources[1][2]
                # save every int(self.run_conf['checkpoint_interval'])
                # TODO: Save the episode_num
                if (episode_num % int(self.run_conf['checkpoint_interval'])) == 0:
                    self.agent.save_checkpoint(self.model_filename)
                    self.memory.save_to_pkl(self.memory_filename)

                #            if self.enable_logging:
                #                self.writer.add_scalar('Episode Reward', episode_reward, t)
                #            episode_rewards.append(episode_reward)
                self.write_tb_values({'reward': episode_reward,
                                      'time': episode_time,
                                      'memory': episode_peak_memory},
                                     episode_num)
                self.episode_results_writer.writerow([episode_num,
                                                      episode_reward,
                                                      episode_time,
                                                      episode_peak_memory])
                self.episode_results_file.flush()
                # s = self.env.reset()[0]
                # episode_done = False
                # episode_reward = 0
                # episode_timesteps = 0
                # episode_num += 1

        self.agent.save_to_onnx(folder=self.run_base_folder_str, critic=False)
        return
