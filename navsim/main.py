from typing import Optional, List, Set

import attr
import cattr
import os
import navsim
from .util import ObjDict
import argparse
import yaml

@attr.s(auto_attribs=True)
class BaseConfig:
    def as_dict(self):
        return cattr.unstructure(self)


class ArgAction(argparse.Action):
    """
    Class to handle actions.
    - Add to the non-default list.
    """
    non_default_args: Set[str] = set()

    def __call__(self, arg_parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        ArgAction.non_default_args.add(self.dest)

class ArgActionStoreTrue(ArgAction):
    """
    Class to handle actions for flags that need to be set to true
    """

    def __init__(self, nargs=0, **kwargs):
        super().__init__(nargs=nargs, **kwargs)

    def __call__(self, arg_parser, namespace, values, option_string=None):
        super().__call__(arg_parser, namespace, True, option_string)

def _create_argparser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "config_file", nargs="?", default=None
    )
    run_conf = argparser.add_argument_group(title="Run Configuration")

    run_conf.add_argument(
        "--run-id",
        default="demo",
        help="The identifier for the training run. This identifier is used to name the "
             "subdirectories in which the trained model and summary statistics are saved as well "
             "as the saved model itself. If you use TensorBoard to view the training statistics, "
             "always set a unique run-id for each training run. (The statistics for all runs with the "
             "same id are combined as if they were produced by a the same session.)",
        action=ArgAction,
    )


    run_conf.add_argument(
        "--train",
        default=True,
        dest="train",
        help="Train the models or just infer",
        action=ArgActionStoreTrue,
    )

    run_conf.add_argument(
        "--debug",
        default=False,
        dest="debug",
        help="Turn debugging on",
        action=ArgActionStoreTrue,
    )

    run_conf.add_argument(
        "--resume",
        default=False,
        dest="resume",
        action=ArgActionStoreTrue,
        help="Whether to resume training or inference from a checkpoint. Specify a --run-id to use this option. "
             "If set, the training code loads an already trained model to initialize the neural network "
             "before resuming training. This option is only valid when the models exist, and have the same "
             "behavior names as the current agents in your scene.",
    )
    run_conf.add_argument(
        "--force",
        default=False,
        dest="force",
        action=ArgActionStoreTrue,
        help="Whether to force-overwrite this run-id's existing summary and model data. (Without "
             "this flag, attempting to train a model with a run-id that has been used before will throw "
             "an error.",
    )

    env_conf = argparser.add_argument_group(title="Environment Configuration")
    env_conf.add_argument(
        "--env",
        default=None,
        dest="env_path",
        help="Path to the Unity executable to train",
        action=ArgAction,
    )

    return argparser


parser = _create_argparser()


@attr.s(auto_attribs=True)
class EnvConfig:
    """
    Configuration for Unity Env
    """
    env_path: Optional[str] = parser.get_default("env_path")
    env_args: Optional[List[str]] = parser.get_default("env_args")
    base_port: int = parser.get_default("base_port")
    num_envs: int = parser.get_default("num_envs")
    seed: int = parser.get_default("seed")

@attr.s(auto_attribs=True)
class RunConfig:
    """
    Configuration for running in general
    """
    run_id: str = parser.get_default("run_id")
    train = parser.get_default("train")
    debug = parser.get_default("num_envs")
    resume = parser.get_default("resume")
    seed = parser.get_default("force")

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
    args = ObjDict(vars(parser.parse_args()))
    print('arguments passed:')
    print(ArgAction.non_default_args)

    # let us load config file first
    config_file = args.config_file or "navsim.yaml"
    try:
        conf = ObjDict().load_from_file(config_file)
    except FileNotFoundError as e:
        if args.config_file is not None:
            abs_path = os.path.abspath(config_file)
            raise OSError(f"Config file could not be found at {abs_path}.") from e
        else:
            print(f"Default config file {config_file} not found, continuing.")
    if conf:
        print(f'Following configuration loaded from {config_file}')
        print(conf.to_yaml())

    # override with CLI args
    if args["env_path"]:
        conf.env_info["env_path"] = args["env_path"]

    print("Final Configuration:")
    print(conf.to_yaml())
    #trainer = navsim.Trainer(run_id=args["run_id"],
    #                         resume=args["resume"],
    #                         conf=conf)

    #trainer.train()
    print("training finished")
    #trainer.env_close()
    print("env closed")
    #trainer.files_close()
    print("files closed")

# For python debugger to directly run this script
if __name__ == "__main__":
    main()
