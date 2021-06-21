from pathlib import Path
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

    run_base_folder = Path(args.run_id)
    run_base_folder_str = str(run_base_folder.resolve())
    if args.resume and run_base_folder.is_dir():
        # TODO: Logic for resume here
        pass
    # else just start fresh
    else:
        args.resume = False
        print("Resume set to True, but nothing to resume from, starting fresh")
        run_base_folder.mkdir(parents=True, exist_ok=True)

    env_conf = ObjDict({
        "log_folder": str((run_base_folder / "unity.log").resolve()),
        "worker_id": 0,
        "base_port": 5005,
        "seed": int(args["seed"]),
        "timeout": int(args["timeout"]),
        "observation_mode": int(args["observation_mode"]),
        "segmentation_mode": int(args["segmentation_mode"]),
        "episode_max_steps": int(args["episode_max_steps"]),
        "task": int(args["task"]),
        "goal": int(args["goal"]),
        "goal_distance": int(args["goal_distance"]),
        "agent_car_physics": int(args["agent_car_physics"]),
        "reward_for_goal": float(args["reward_for_goal"]),
        "reward_for_ep": float(args["reward_for_ep"]),
        "reward_for_other": float(args["reward_for_other"]),
        "reward_for_falling_off_map": float(args["reward_for_falling_off_map"]),
        "reward_for_step": float(args["reward_for_step"]),
        "env_path": args["env_path"],
        "debug": args["debug"],
        "run_base_folder_str": run_base_folder_str
    })

    run_conf = ObjDict({
        "run_id": args["run_id"],
        "env_name": "navsim",
        "episode_max_steps": int(args["episode_max_steps"]),
        "total_episodes": int(args["total_episodes"]),
        "seed": int(args["seed"]),
        "discount": float(args["discount"]),
        "tau": float(args["tau"]),
        "expl_noise": float(args["exploration_noise"]),
        "memory_capacity": int(args["memory_capacity"]),
        "batch_size": int(args["batch_size"]),
        "batches_before_train": int(args["batches_before_train"]),
        "checkpoint_interval": int(args["checkpoint_interval"]),
        "train_interval": int(args["train_interval"])
    })

    if args["rl_backend"] == "rllib":
        import ray.rllib.agents.ppo as ppo
        conf = ppo.DEFAULT_CONFIG.copy()
        if args.debug:
            conf["log_level"] = "DEBUG"
        else:
            conf["log_level"] = "INFO"

        conf["framework"] = "torch"
        conf["ignore_worker_failures"] = True
    else:
        conf = ObjDict()
        conf['run_config'] = run_conf

    conf["env"] = "navsim"
    conf["num_workers"] = 1
    conf['env_config'] = env_conf

    print("Passed arguments + defaults:")
    print(args.to_yaml())
    # print("Final Configuration:")
    # print(conf.to_yaml())

    if args["rl_backend"] == "rllib":
        import ray
        ray.shutdown()
        ray.init(ignore_reinit_error=True)
        navsim.NavSimGymEnv.register_with_ray()
        result = ray.tune.run(
            ppo.PPOTrainer,
            config=conf,
            name=run_conf.run_id,
            resume=args.resume,
            local_dir=run_base_folder_str,
            stop={"episodes_total": run_conf.total_episodes},
            checkpoint_freq=run_conf.checkpoint_interval,
            checkpoint_at_end=True
        )
        best_checkpoint = result.get_last_checkpoint(
            metric="episode_reward_mean", mode="max"
        )
        # best_checkpoint = result.get_trial_checkpoints_paths(
        #    trial=result.get_best_trial("episode_reward_mean"),
        #    metric="episode_reward_mean", mode="max")

        print(best_checkpoint)
        trainer = ppo.PPOTrainer(config=conf)
        trainer.restore(best_checkpoint)
        model = trainer.get_policy().model
        print(type(model))
        print(model)
    else:
        # if resume is passed then read the args from saved conf instead and then overwrite with thhe args passed
        # TODO: Optimize this with respect to above arg/config setting
        if args.resume and run_base_folder.is_dir():
            conf = ObjDict().load_from_json_file(f"{run_base_folder_str}/conf"
                                                 f".json")
            for passed_arg in non_default_args:
                if passed_arg in conf["env_config"]:
                    conf["env_config"][passed_arg] = args[passed_arg]
                if passed_arg in conf["run_config"]:
                    conf["run_config"][passed_arg] = args[passed_arg]
                if passed_arg in conf:
                    conf[passed_arg] = args[passed_arg]

            for arg in ["seed", "memory_capacity", "batch_size",
                        "batches_before_train", "checkpoint_interval",
                        "train_interval"]:
                conf["run_config"][arg] = int(conf["run_config"][arg])
            for arg in ["discount", "tau", "expl_noise"]:
                conf["run_config"][arg] = float(conf["run_config"][arg])
            for arg in ["seed", "timeout", "base_port", "observation_mode",
                        "segmentation_mode", "episode_max_steps", "task",
                        "goal","goal_distance", "agent_car_physics"]:
                conf["env_config"][arg] = int(conf["env_config"][arg])
            for arg in ["reward_for_goal", "reward_for_ep", "reward_for_other",
                        "reward_for_falling_off_map", "reward_for_step"]:
                conf["env_config"][arg] = float(conf["env_config"][arg])

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
