import csv
import cv2
from pathlib import Path
import json

import torch
from tqdm import tqdm
import numpy as np
import math

import gym
from navsim.agent import DDPGAgent
from navsim.util import sizeof_fmt, image_layout, s_hwc_to_chw
from navsim.util import ObjDict, ResourceCounter

import traceback
from torch.utils.tensorboard import SummaryWriter


class TorchJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        # elif isinstance(obj, np.floating):
        #    return float(obj)
        # elif isinstance(obj, np.ndarray):
        #    return obj.tolist()
        else:
            return super(TorchJSONEncoder, self).default(obj)


"""
def s_hwc_to_chw(s):
    # TODO: HWC to CHW conversion optimized here
    # because pytorch can only deal with images in CHW format
    # we are making the optimization here to convert from HWC to CHW format.
    if isinstance(s, np.ndarray) and (s.ndim > 2):
        s = image_layout(s, 'hwc', 'chw')
    elif isinstance(s, list):  # state is a list of states
        for i in range(len(s)):
            if isinstance(s[i], np.ndarray) and (s[i].ndim > 2):
                s[i] = image_layout(s[i], 'hwc', 'chw')
    return s
"""


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

        #self.conf = conf
        self.resume = resume

        self.run_base_folder = Path(self.run_id).resolve()
        self.run_base_folder_str = str(self.run_base_folder)
        #        if run_base_folder.is_dir():
        #            raise ValueError(f"{run_base_folder_str} exists as a non-directory. "
        #                             f"Please remove the file or use a different run_id")
        if resume and self.run_base_folder.is_dir():
            # self.conf = ObjDict().load_from_json_file(f"{self.run_base_folder_str}/conf.json")
            self.file_mode = 'a+'

        # else just start fresh
        else:

            self.run_base_folder.mkdir(parents=True, exist_ok=True)
            conf.save_to_json_file(f"{self.run_base_folder_str}/conf.json")
            self.file_mode = 'w+'

        self.run_conf = ObjDict(conf.run_config)
        self.env_config = ObjDict(conf.env_config)

        pylog_filename = self.run_base_folder / 'py.log'  # TODO: use logger
        self.pylog_filename = str(pylog_filename)
        step_results_filename = self.run_base_folder / 'step_results.csv'
        self.step_results_filename = str(step_results_filename)
        episode_results_filename = self.run_base_folder / 'episode_results.csv'
        self.episode_results_filename = str(episode_results_filename)

        self.episode_num_filename = f"{self.run_base_folder_str}/last_checkpointed_episode_num.txt"

        # TODO: Add the code to delete previous files
        # TODO: Add the code to add categories
        env_log_folder = self.run_base_folder / 'env_log'
        tb_folder = self.run_base_folder / 'tb'
        agent_folder = self.run_base_folder / 'agent'
        if not resume:
            import shutil
            for folder in [tb_folder, agent_folder, env_log_folder]:
                if folder.exists():
                    shutil.rmtree(folder)
                folder.mkdir(parents=True, exist_ok=True)

        self.agent_folder_str = str(agent_folder)
        self.env_config["log_folder"] = str(env_log_folder)
        self.summary_writer = SummaryWriter(f"{str(tb_folder)}")

        self.rc = ResourceCounter()
        self.files_open()

        try:
            if resume and self.run_base_folder.is_dir():
                with open(self.episode_num_filename,
                          mode='r') as episode_num_file:
                    episode_num = int(episode_num_file.read())
                    self.env_config["start_from_episode"] = episode_num + 1
            else:
                self.env_config["start_from_episode"] = 1
            self.env = None
            self.env_open()

            self.max_action = self.env.action_space.high[0]

            # Let us set the GPU or CPU

            if torch.cuda.is_available():
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
                self.device = torch.device(
                    f"cuda:{self.run_conf['agent_gpu_id']}")
                torch.cuda.set_device(self.run_conf['agent_gpu_id'])
                torch.cuda.empty_cache()
            else:
                self.device = torch.device('cpu')
                torch.set_num_threads(8)
                torch.set_num_interop_threads(8)
            print(f'Torch is using {self.device}, '
                  f'{torch.get_num_threads()} threads and '
                  f'{torch.get_num_interop_threads()} interop threads')

            if self.run_conf["mem_backend"] == "cupy":
                if torch.cuda.is_available():
                    from navsim.memory import CupyMemory
                    CupyMemory.set_device(self.run_conf['agent_gpu_id'])
                    mem_backend = CupyMemory
                else:
                    raise ValueError(
                        "mem_backend=cupy but GPU is not available")
            elif self.run_conf["mem_backend"] == "numpy":
                if torch.cuda.is_available():
                    print("Warning: GPU is available but mem_backend=numpy")
                from navsim.memory import NumpyMemory
                mem_backend = NumpyMemory

            if resume and self.run_base_folder.is_dir():
                model_filename = f"{self.agent_folder_str}/{episode_num}_model_state.pt"
                memory_filename = f"{self.agent_folder_str}/{episode_num}_memory.pkl"
                self.memory = mem_backend.load_from_pkl(memory_filename)
            else:
                memory_observation_space_shapes = []
                for item in self.env.observation_space_shapes:
                    if len(item) == 3:
                        # means its an image in HWC
                        # we pass the shapes as CWH
                        memory_observation_space_shapes.append(
                            (item[2], item[1], item[0]))
                    else:
                        memory_observation_space_shapes.append(item)
                self.memory = mem_backend(
                    capacity=self.run_conf.memory_capacity,
                    state_shapes=memory_observation_space_shapes,
                    action_shape=self.env.action_space.shape,
                    seed=self.run_conf['seed']
                )
            self.memory.info()

            self.agent = DDPGAgent(
                env=self.env,
                device=self.device,
                discount=self.run_conf['discount'], tau=self.run_conf['tau']
            )
            if resume and self.run_base_folder.is_dir():
                self.agent.load_checkpoint(model_filename)
            # TODO: self.agent.info()

            dummy_obs = [torch.as_tensor(obs,
                                         dtype=torch.float) for obs in
                         s_hwc_to_chw(self.env.get_dummy_obs()[0])]
            dummy_act = torch.as_tensor(self.env.get_dummy_actions()[0],
                                        dtype=torch.float
                                        )

            dummy_nn = self.agent.NN_WRAPPER(self.env.observation_space_shapes,
                                             self.env.action_space.shape[0],
                                             self.env.action_space.high[0])
            self.summary_writer.add_graph(dummy_nn,
                                          [dummy_obs, dummy_act]
                                          )
            del dummy_nn, dummy_obs, dummy_act

            torch.manual_seed(self.run_conf['seed'])
            np.random.seed(self.run_conf['seed'])

        except Exception as e:
            self.env_close()
            self.files_close()
            print(traceback.format_exc())
            # print(e)

    # TODO: Find a better name for  this function
    def write_tb(self, group, values, t: int):
        for key, value in values.items():
            self.summary_writer.add_scalar(f'{group}/{key}', value, t)
            self.summary_writer.flush()

    def files_open(self):
        self.pylog_file = open(self.pylog_filename, mode=self.file_mode)

        self.step_results_file = open(self.step_results_filename,
                                      mode=self.file_mode)
        self.step_results_writer = csv.writer(self.step_results_file,
                                              delimiter=',',
                                              quotechar='"',
                                              quoting=csv.QUOTE_MINIMAL)

        self.episode_results_file = open(self.episode_results_filename,
                                         mode=self.file_mode)
        self.episode_results_writer = csv.writer(self.episode_results_file,
                                                 delimiter=',',
                                                 quotechar='"',
                                                 quoting=csv.QUOTE_MINIMAL)
        if not self.resume:
            self.step_results_writer.writerow(
                ['episode_num', 't', 'r', 'step_time', 'env_step_time',
                 'memory_append_time', 'memory_sample_time',
                 'agent_train_time', 'current_memory'])

            self.episode_results_writer.writerow(
                ['episode_num',
                 'episode_reward',
                 'episode_time',
                 'episode_peak_memory'])

        self.step_results_file.flush()
        self.episode_results_file.flush()

    def files_close(self):
        self.pylog_file.close()
        self.step_results_file.close()
        self.episode_results_file.close()

    def env_open(self):
        self.rc.start()

        self.env = gym.make(self.run_conf.env, env_config=self.env_config)
        time_since_start, current_memory, peak_memory = self.rc.stop()
        log_str = f'Unity env creation resource usage: \n' \
                  f'time:{time_since_start},' \
                  f'peak_memory:{sizeof_fmt(peak_memory)},' \
                  f'current_memory:{sizeof_fmt(current_memory)}\n'
        self.pylog_file.write(log_str)
        print(log_str)
        self.pylog_file.flush()
        self.env.info()
        # self.env.info_steps(save_visuals=True)

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
        total_episodes = int(self.run_conf['total_episodes'])
        num_episodes = total_episodes - (
                self.env_config["start_from_episode"] - 1)
        train_interval = int(self.run_conf['train_interval'])
        checkpoint_interval = int(self.run_conf['checkpoint_interval'])
        num_episode_blocks = int(math.ceil(num_episodes / checkpoint_interval))

        batch_size = self.run_conf['batch_size']
        # save the state json at start of run
        model_filename = f"{self.run_base_folder_str}/" \
                         f"{self.env_config['start_from_episode']}_" \
                         f"{total_episodes}_start_agent_state.json"
        json.dump(self.agent.get_state_dict(), open(model_filename, 'w'),
                  indent=2, sort_keys=True, cls=TorchJSONEncoder)

        #        print("Debugging training loop")
        #        print(f"{num_episode_blocks},{num_episodes},{self.conf.env_config['start_from_episode']}")
        t_global = 1

        episode_resources = np.full((checkpoint_interval, 2, 3), 0.0)
        # [[[0] * 3] * 2] * checkpoint_interval
        episode_steps = [0] * checkpoint_interval
        step_res = np.full((checkpoint_interval, t_max, 10, 3), 0.0)
        # [[[[0] * 3] * 10] * t_max] * checkpoint_interval
        step_rew = np.full((checkpoint_interval, t_max), 0.0)
        step_spl = np.full((checkpoint_interval, t_max), 0.0)
        step_loss = np.full((checkpoint_interval, t_max, 2), 0.0)

        for i in range(0, num_episode_blocks):
            start_episode = (self.env_config["start_from_episode"] - 1) + (
                    (i * checkpoint_interval) + 1)
            stop_episode = min((self.env_config["start_from_episode"] - 1)
                               + ((i + 1) * checkpoint_interval),
                               total_episodes)
            episodes_in_block = stop_episode - start_episode + 1
            # episode_rewards = np.full(episodes_in_block, 0.0)
            ckpt_e = 0
            ckpt_t_global = t_global

            for episode_num in tqdm(
                    iterable=range(start_episode, stop_episode + 1),
                    desc=f"Episode {start_episode}-{stop_episode}/{total_episodes}",
                    unit='episode', ascii=True, ncols=80,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]'):
                self.rc.start()
                episode_counter = episode_num - start_episode
                episode_resources[ckpt_e, 0] = self.rc.snapshot()  # e0

                # initialise the episode counters
                episode_done = False
                # episode_return = 0
                t = 1

                # self.env.start_navigable_map()
                # observe initial s
                s = self.env.reset()  # s comes out of env as a tuple always
                # because pytorch can only deal with images in CHW format
                # we are making the optimization here to convert from HWC to CHW format.
                s = s_hwc_to_chw(s)

                # self.env.step(self.env.action_space.sample())
                # get the navigable map and save it as image
                # navigable_map = self.env.get_navigable_map()
                # if navigable_map is not None:
                #    cv2.imwrite(str((self.run_base_folder / f'navigable_map_{episode_num}.jpg').resolve()),navigable_map*255)
                # else:
                #    print(f'Map for episode {episode_num} is None')
                # step_res = [[0, 0, 0]] * 10

                while not episode_done:
                    # s0: start of overall_step
                    # s1,2: before and after memory.sample()
                    # s3,4: before and after agent.train()
                    # s5,6: before and after env.step()
                    # s7,8: before and after memory.append()
                    # s9: end of step
                    step_res[ckpt_e, t - 1, 0] = self.rc.snapshot()  # s0

                    # do the random sampling until enough memory is full
                    if self.memory.size < batch_size:
                        a = self.env.action_space.sample()
                        # rescale break between 0,1 from -1,1
                        # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
                        # a[2] = (a[2] + 1) / 2
                        step_res[ckpt_e, t - 1, 1:5] = [[0.0] * 3] * 4
                        # s1-4
                    else:
                        # TODO: Find the best place to train, moved here for now
                        if train_interval and (
                                (t_global % train_interval) == 0):
                            step_res[ckpt_e, t - 1, 1] = self.rc.snapshot()
                            # s1
                            batch_s, batch_a, batch_r, batch_s_, batch_d = \
                                self.memory.sample(batch_size)
                            step_res[ckpt_e, t - 1, 2] = self.rc.snapshot()
                            # s2
                            # print('training the agent')
                            step_res[ckpt_e, t - 1, 3] = self.rc.snapshot()
                            # s3
                            self.agent.train(batch_s, batch_a, batch_r,
                                             batch_s_, batch_d)
                            step_res[ckpt_e, t - 1, 4] = self.rc.snapshot()
                            # s4
                        else:
                            step_res[ckpt_e, t - 1, 1:5] = [[0.0] * 3] * 4
                            # s1-4

                        a = (self.agent.select_action(s) + np.random.normal(
                            0, self.max_action * self.run_conf['expl_noise'],
                            size=self.env.action_space.shape[0])
                             ).clip(
                            -self.max_action,
                            self.max_action
                        )

                    step_res[ckpt_e, t - 1, 5] = self.rc.snapshot()  # s5
                    s_, r, episode_done, info = self.env.step(a)
                    step_res[ckpt_e, t - 1, 6] = self.rc.snapshot()  # s6

                    # because pytorch can only deal with images in CHW format
                    # we are making the optimization here to convert from HWC to CHW format.
                    s_ = s_hwc_to_chw(s_)

                    #    s = [[s]]  # make state a list of sequences
                    #    s_ = [[s_]]
                    # elif isinstance(s[0], (int,float)):  # state does not have multiple seq
                    # s = [s]
                    # s_ = [s_]
                    # for item in s_:

                    step_res[ckpt_e, t - 1, 7] = self.rc.snapshot()
                    # s7
                    self.memory.append(
                        s=s, a=a, s_=s_, r=r,
                        d=float(episode_done))  # if t < t_max -1 else 1)
                    s = s_
                    step_res[ckpt_e, t - 1, 8] = self.rc.snapshot()  # s8

                    step_rew[ckpt_e, t - 1] = r
                    step_spl[ckpt_e, t - 1] = self.env.spl_current
                    step_loss[ckpt_e, t - 1] = [
                        0 if self.agent.actor_loss is None else self.agent.actor_loss.data.cpu().numpy(),
                        0 if self.agent.critic_loss is None else self.agent.critic_loss.data.cpu().numpy()]

                    # print(step_loss[ckpt_e, t - 1])
                    # episode_rewards[ckpt_e] += r

                    # if self.memory.size >= self.run_conf['batch_size'] * self.run_conf['batches_before_train']:

                    #                    if (t >= self.config['batch_size'] * self.config['batches_before_train']) and (t % 1000 == 0):
                    # episode_evaluations.append(evaluate_policy(self.agent, self.env, self.config['seed']))

                    step_res[ckpt_e, t - 1, 9] = self.rc.snapshot()  # s9
                    # print(ckpt_e,t-1, step_res[ckpt_e][t-1])
                    # print(t,step_res[0][0])
                    if episode_done or (t_max and (t >= t_max)):
                        break
                    else:
                        t += 1
                        t_global += 1
                # print(step_res[0][0])
                # end of while loop for one episode
                # episode end processing
                episode_resources[ckpt_e, 1] = self.rc.stop()  # e1
                episode_steps[ckpt_e] = t
                ckpt_e += 1

                #            if self.enable_logging:
                #                self.writer.add_scalar('Episode Reward', episode_return, t)
                #            episode_rewards.append(episode_return)

                # s = self.env.reset()[0]
                # episode_done = False
                # episode_return = 0
                # episode_timesteps = 0
                # episode_num += 1
            # end of tqdm-for loop - checkpoint block of episodes
            # now lets checkpoint everything

            t_global = ckpt_t_global
            # model and memory checkpoint
            with open(self.episode_num_filename, mode='w') as episode_num_file:
                episode_num_file.write(str(episode_num))
            model_filename = f"{self.agent_folder_str}/{episode_num}_model_state.pt"
            memory_filename = f"{self.agent_folder_str}/{episode_num}_memory.pkl"
            self.agent.save_checkpoint(model_filename)
            self.memory.save_to_pkl(memory_filename)

            # print(step_res[0][0])
            # episode data checkpoint
            for e_num in range(episodes_in_block):
                episode_time = episode_resources[e_num, 1, 0] - \
                               episode_resources[e_num, 0, 0]
                episode_return = step_rew[e_num, 0:episode_steps[e_num]].sum()
                self.write_tb('episode',
                              {'return': episode_return,
                               'time': episode_time,
                               'peak_memory': episode_resources[e_num, 1, 2],
                               'total_steps': episode_steps[e_num],
                               'max_spl': step_spl[e_num].max(),
                               'min_spl': step_spl[e_num].min(),
                               'spl_span': step_spl[e_num].max() - step_spl[
                                   e_num].min()
                               },
                              start_episode + e_num)  # episode_num
                self.episode_results_writer.writerow(
                    [start_episode + e_num,  # episode_num
                     episode_return,
                     episode_time,
                     episode_resources[e_num, 1, 2]])  # episode_peak_memory

                # step data checkpoint
                for t in range(episode_steps[e_num]):
                    # s0: start of overall_step
                    # s1,2: before and after memory.sample()
                    # s3,4: before and after agent.train()
                    # s5,6: before and after env.step()
                    # s7,8: before and after memory.append()
                    # s9: end of step
                    # print(e_num,t,step_res[e_num][t])
                    step_time = step_res[e_num, t, 9, 0] - step_res[
                        e_num, t, 0, 0]
                    memory_sample_time = step_res[e_num, t, 2, 0] - step_res[
                        e_num, t, 1, 0]
                    agent_train_time = step_res[e_num, t, 4, 0] - step_res[
                        e_num, t, 3, 0]
                    env_step_time = step_res[e_num, t, 6, 0] - step_res[
                        e_num, t, 5, 0]
                    memory_append_time = step_res[e_num, t, 8, 0] - step_res[
                        e_num, t, 7, 0]
                    # peak_memory = step_res[3][2]
                    current_memory = step_res[e_num, t, 9, 1]
                    r = step_rew[e_num, t]
                    self.write_tb('step',
                                  {'step_reward': r,
                                   'step_spl': step_spl[e_num, t]
                                   },
                                  t_global)
                    self.write_tb('time',
                                  {
                                      'step_time': step_time,
                                      'env_step_time': env_step_time,
                                      'memory_append_time': memory_append_time,
                                      'memory_sample_time': memory_sample_time,
                                      'agent_train_time': agent_train_time,
                                      'current_memory': current_memory,
                                  },
                                  t_global)
                    self.write_tb('loss',
                                  {
                                      'actor_loss': step_loss[e_num, t, 0],
                                      'critic_loss': step_loss[e_num, t, 1],
                                  },
                                  t_global)
                    self.step_results_writer.writerow(
                        [start_episode + e_num, t + 1, r, step_time,
                         env_step_time,
                         memory_append_time, memory_sample_time,
                         agent_train_time, current_memory])
                    t_global += 1
                # steps checkpoint loop finishes
            # episodes checkpoint loop finishes
            self.episode_results_file.flush()
            self.step_results_file.flush()
            ckpt_e = 0
        # save the state json at end of run
        model_filename = f"{self.run_base_folder_str}/" \
                         f"{self.env_config['start_from_episode']}_" \
                         f"{total_episodes}_stop_agent_state.json"
        json.dump(self.agent.get_state_dict(), open(model_filename, 'w'),
                  indent=2, sort_keys=True, cls=TorchJSONEncoder)

        self.agent.save_to_onnx(folder=self.run_base_folder_str, critic=False)
        return
