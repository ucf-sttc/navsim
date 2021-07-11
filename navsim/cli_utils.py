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

    run_desc = "The arguments are used to configure the runtime execution."
    run_conf = argparser.add_argument_group(title="Run Configuration",
                                            description=run_desc)

    run_conf.add_argument(
        "--run_id",
        default="demo",
        help="The identifier for the training run. This identifier is used to name the "
             "subdirectories in which the trained model and summary statistics are saved as well "
             "as the saved model itself. If you use TensorBoard to view the training statistics, "
             "always set a unique run-id for each training run. (The statistics for all runs with the "
             "same id are combined as if they were produced by a the same session.)",
        action=ArgAction,
    )

    run_conf.add_argument(
        "--rl_backend",
        default=None,
        help="The backend library for RL.",
        action=ArgAction,
    )

    run_conf.add_argument(
        "--debug",
        default=False,
        help="Turn debugging on",
        action=ArgActionStoreTrue,
    )

    run_conf.add_argument(
        "--resume",
        default=False,
        action=ArgActionStoreTrue,
        help="Whether to resume training or inference from a checkpoint. Specify a --run-id to use this option. "
             "If set, the training code loads an already trained model to initialize the neural network "
             "before resuming training. This option is only valid when the models exist, and have the same "
             "behavior names as the current agents in your scene.",
    )
    run_conf.add_argument(
        "--force",
        default=False,
        action=ArgActionStoreTrue,
        help="Whether to force-overwrite this run-id's existing summary and model data. (Without "
             "this flag, attempting to train a model with a run-id that has been used before will throw "
             "an error.",
    )

    run_conf.add_argument(
        "--total_episodes",
        default=2,
        action=ArgAction,
        help="Total number of episodes to run. If resume is used, then it will try to read the previously run episodes and continue from there.",
    )
    run_conf.add_argument(
        "--train_interval",
        default=16,
        help="Train the model after these many global steps. If set to 0 then model wont train",
        action=ArgAction,
    )

    run_conf.add_argument(
        "--checkpoint_interval",
        default=1,
        action=ArgAction,
        help="Execute the episodes in blocks of checkpoint intervals",
    )

    run_conf.add_argument(
        "--agent_gpu_id",
        default=0,
        action=ArgAction,
        help="Which GPU to run models in agent on",
    )

    run_conf.add_argument(
        "--episode_max_steps",
        default=100,
        action=ArgAction,
        help="Maximum number of steps in an Episode, aka Episode Length",
    )

    run_conf.add_argument(
        "--seed",
        default=0,
        action=ArgAction,
        help="Seed",
    )

    run_conf.add_argument(
        "--discount",
        default=0.99,
        action=ArgAction,
        help="discount",
    )
    run_conf.add_argument(
        "--tau",
        default=5e-3,
        action=ArgAction,
        help="tau",
    )
    run_conf.add_argument(
        "--exploration_noise",
        default=0.1,
        action=ArgAction,
        help="exploration_noise",
    )
    run_conf.add_argument(
        "--batch_size",
        default=32,
        action=ArgAction,
        help="Batch Size",
    )

    run_conf.add_argument(
        "--memory_capacity",
        default=100,
        action=ArgAction,
        help="Total capacity of memory, should be > batch_size",
    )

    env_desc = "The arguments are used to configure the environment."
    env_conf = argparser.add_argument_group(title="Environment Configuration",
                                            description=env_desc)
    env_conf.add_argument(
        "--env",
        default=None,
        dest="env_path",
        help="Path to the Unity executable to train",
        action=ArgAction,
    )

    env_conf.add_argument(
        "--env_gpu_id",
        default=0,
        action=ArgAction,
        help="Which GPU to run env on",
    )
    env_conf.add_argument(
        "--timeout",
        default=600,
        help="TimeOut for the Env",
        action=ArgAction,
    )
    env_conf.add_argument(
        "--obs_mode",
        default=0,
        help="Observation Mode : 0,1,2",
        action=ArgAction,
    )

    env_conf.add_argument(
        "--obs_height",
        default=256,
        help="Observation height",
        action=ArgAction,
    )

    env_conf.add_argument(
        "--obs_width",
        default=256,
        help="Observation width",
        action=ArgAction,
    )
    env_conf.add_argument(
        "--segmentation_mode",
        default=1,
        help="Segmentation Mode : 1",
        action=ArgAction,
    )
    env_conf.add_argument(
        "--task",
        default=0,
        help="Task",
        action=ArgAction,
    )
    env_conf.add_argument(
        "--goal",
        default=0,
        help="Goal",
        action=ArgAction,
    )
    env_conf.add_argument(
        "--goal_distance",
        default=50,
        help="Distance to goal from current location",
        action=ArgAction,
    )
    env_conf.add_argument(
        "--agent_car_physics",
        default=0,
        help="Agent Car Physics Levels : 0,1,2,10",
        action=ArgAction,
    )
    env_conf.add_argument(
        "--reward_for_goal",
        default=50,
        help="Reward for reaching the goal",
        action=ArgAction,
    )
    env_conf.add_argument(
        "--reward_for_no_viable_path",
        default=-50,
        help="Reward for no more viable paths remaining to reach the goal",
        action=ArgAction,
    )
    env_conf.add_argument(
        "--reward_step_mul",
        default=0.1,
        help="Multiply step reward [which is -(goal_reward / spl_start) with this number",
        action=ArgAction,
    )
    env_conf.add_argument(
        "--reward_collision_mul",
        default=4,
        help="Multiply step reward with this number when there is a collision and add to reward",
        action=ArgAction,
    )

    env_conf.add_argument(
        "--reward_spl_delta_mul",
        default=1,
        help="Shortest path length delta multiplier",
        action=ArgAction,
    )

    env_conf.add_argument(
        "--save_visual_obs",
        default=False,
        action=ArgActionStoreTrue,
        help="Save the visual observations at every step",
    )
    env_conf.add_argument(
        "--save_vector_obs",
        default=False,
        action=ArgActionStoreTrue,
        help="Save the vector observations at every step",
    )

    dev_desc = "The arguments are used by developers to benchmark and debug navsim API"
    dev_conf = argparser.add_argument_group(title="Developer Configuration",
                                            description=dev_desc)

    dev_conf.add_argument(
        "--mem_backend",
        default="cupy",
        help="The backend library for Rollback Memory.",
        action=ArgAction,
    )

    return argparser


argparser = _create_argparser()
