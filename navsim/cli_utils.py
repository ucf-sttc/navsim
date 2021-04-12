import argparse
from typing import Optional, List, Set

non_default_args: Set[str] = set()

class ArgAction(argparse.Action):
    """
    Class to handle actions.
    - Add to the non-default list.
    """
    def __call__(self, arg_parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        non_default_args.add(self.dest)


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
    # argparser.add_argument(
    #    "config_file", nargs="?", default=None
    # )
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

    run_conf.add_argument(
        "--num_episodes",
        default=2,
        dest="num_episodes",
        action=ArgAction,
        help="",
    )

    run_conf.add_argument(
        "--episode_max_steps",
        default=10,
        dest="episode_max_steps",
        action=ArgAction,
        help="",
    )

    run_conf.add_argument(
        "--seed",
        default=123,
        dest="seed",
        action=ArgAction,
        help="",
    )

    run_conf.add_argument(
        "--discount",
        default=0.99,
        dest="discount",
        action=ArgAction,
        help="",
    )
    run_conf.add_argument(
        "--tau",
        default=5e-3,
        dest="tau",
        action=ArgAction,
        help="",
    )
    run_conf.add_argument(
        "--exploration_noise",
        default=0.1,
        dest="exploration_noise",
        action=ArgAction,
        help="",
    )
    run_conf.add_argument(
        "--batch_size",
        default=256,
        dest="batch_size",
        action=ArgAction,
        help="",
    )
    run_conf.add_argument(
        "--checkpoint_interval",
        default=1,
        dest="checkpoint_interval",
        action=ArgAction,
        help="",
    )

    env_conf = argparser.add_argument_group(title="Environment Configuration")
    env_conf.add_argument(
        "--env",
        default=None,
        dest="env_path",
        help="Path to the Unity executable to train",
        action=ArgAction,
    )

    env_conf.add_argument(
        "--goal_distance",
        default=50,
        dest="goal_distance",
        help="DIstance to goal from current location",
        action=ArgAction,
    )

    return argparser


argparser = _create_argparser()