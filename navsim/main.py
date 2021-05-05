from typing import Optional, List, Set

import attr
import cattr
import os
import navsim
from .util import ObjDict
import argparse
import yaml
from .cli_utils import argparser, non_default_args

@attr.s(auto_attribs=True)
class RunConfig:
    """
    Configuration for running in general
    """
    run_id: str = argparser.get_default("run_id")
    train = argparser.get_default("train")
    debug = argparser.get_default("num_envs")
    resume = argparser.get_default("resume")
    seed = argparser.get_default("force")


def main():
    """
    TODO: Implement configuration checks
    :return:
    """
    print(f'===================================')
    print(f'Navsim Version {navsim.__version__}')
    print(f'===================================')
    args = ObjDict(vars(argparser.parse_args()))
    print('arguments passed:')
    print(non_default_args)

    env_conf = ObjDict({
        "log_folder": "unity.log",
        "seed": 123,
        "timeout": 600,
        "worker_id": 0,
        "base_port": 5005,
        "observation_mode": 2,
        "segmentation_mode": 1,
        "max_steps": int(args["episode_max_steps"])+2,
        "task": 0,
        "goal": 0,
        "goal_distance": int(args["goal_distance"]),
        "reward_for_goal": 50,
        "reward_for_ep": 0.005,
        "reward_for_other": -0.1,
        "reward_for_falling_off_map": -50,
        "reward_for_step": -0.0001,
        "agent_car_physics": 0,
        "episode_max_steps": 10,
        "env_path":args["env_path"]
    })

    run_conf = ObjDict({
        "env_name": "navsim",
        "episode_max_steps": int(args["episode_max_steps"]),
        "num_episodes": int(args["num_episodes"]),
        "seed": int(args["seed"]),
        "discount": float(args["discount"]),
        "tau": float(args["tau"]),
        "expl_noise": float(args["exploration_noise"]),
        "memory_capacity": 100,
        "batch_size": int(args["batch_size"]),
        "batches_before_train": 2,
        "checkpoint_interval": int(args["checkpoint_interval"])
    })
    run_conf["run_id"]=args["run_id"]

    conf = ObjDict({
        'env_conf': env_conf, 'run_conf': run_conf
    })
    print("Passed arguments + defaults:")
    print(args.to_yaml())
    print("Final Configuration:")
    print(conf.to_yaml())
    executor = navsim.Executor(run_id=args["run_id"],
                               resume=args["resume"],
                               conf=conf)

    executor.execute()
    print("training finished")

    executor.env_close()
    print("env closed")
    executor.files_close()
    print("files closed")


# For python debugger to directly run this script
if __name__ == "__main__":
    main()
