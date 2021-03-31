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
class BaseConfig:
    def as_dict(self):
        return cattr.unstructure(self)

@attr.s(auto_attribs=True)
class EnvConfig:
    """
    Configuration for Unity Env
    """
    env_path: Optional[str] = argparser.get_default("env_path")
    env_args: Optional[List[str]] = argparser.get_default("env_args")
    base_port: int = argparser.get_default("base_port")
    num_envs: int = argparser.get_default("num_envs")
    seed: int = argparser.get_default("seed")


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


@attr.s(auto_attribs=True)
class Config(BaseConfig):
    """
    # Options: command line -> yaml_file -> defaults

    """
    env_config: EnvConfig = attr.ib(factory=EnvConfig)
    run_config: EnvConfig = attr.ib(factory=EnvConfig)

    @staticmethod
    def from_args_dict(args):
        # create config dict from command line default values
        # If config file given in command line
        # if the config file doesnt exist then error and exit
        # else read config and overwrite default values
        # add command line values
        # let us load config file first
        config_file = args.config_file
        if config_file:
            try:
                conf = ObjDict().load_from_file(config_file)
            except FileNotFoundError as e:
                if args.config_file is not None:
                    abs_path = os.path.abspath(config_file)
                    raise OSError(f"Config file could not be found at {abs_path}.") from e
            if conf:
                print(f'Following configuration loaded from {config_file}')
                print(conf.to_yaml())
        else:
            print(f"Config file not specified in command line, continuing.")

        return True


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
        "observation_mode": 0,
        "segmentation_mode": 1,
        "max_steps": 10,
        "task": 0,
        "goal": 0,
        "reward_for_goal": 50,
        "reward_for_ep": 0.005,
        "reward_for_other": -0.1,
        "reward_for_falling_off_map": -50,
        "reward_for_step": -0.0001,
        "agent_car_physics": 0,
        "episode_max_steps": 10,
    })
    env_conf["env_path"]=args["env_path"]

    run_conf = ObjDict({
        "env_name": "navsim",
        "episode_max_steps": args["episode_max_steps"],
        "num_episodes": args["num_episodes"],
        "seed": args["seed"],
        "discount": args["discount"],
        "tau": args["tau"],
        "expl_noise": args["exploration_noise"],
        "memory_capacity": 100,
        "batch_size": args["batch_size"],
        "batches_before_train": 2,
        "checkpoint_interval": args["checkpoint_interval"]
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
