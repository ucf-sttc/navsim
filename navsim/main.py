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
        #TODO: Logic for resume here
        pass
    # else just start fresh
    else:
        run_base_folder.mkdir(parents=True, exist_ok=True)

    env_conf = ObjDict({
        "log_folder": str((run_base_folder / "unity.log").resolve()),
        "seed": int(args["seed"]),
        "timeout": 600,
        "worker_id": 0,
        "base_port": 5005,
        "observation_mode": int(args["observation_mode"]),
        "segmentation_mode": 1,
        "episode_max_steps": int(args["episode_max_steps"]) + 2,
        "task": 0,
        "goal": 0,
        "goal_distance": int(args["goal_distance"]),
        "reward_for_goal": 50,
        "reward_for_ep": 0.005,
        "reward_for_other": -0.1,
        "reward_for_falling_off_map": -50,
        "reward_for_step": -0.0001,
        "agent_car_physics": 0,
        "env_path": args["env_path"],
        "debug":args["debug"],
        "run_base_folder_str":run_base_folder_str
    })

    run_conf = ObjDict({
        "run_id": args["run_id"],
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
        "checkpoint_interval": int(args["checkpoint_interval"]),
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
    #print("Final Configuration:")
    #print(conf.to_yaml())

    if args["rl_backend"] == "rllib":
        import ray
        ray.shutdown()
        ray.init(ignore_reinit_error=True)
        navsim.NavSimGymEnv.register_with_ray()
        result = ray.tune.run(
            ppo.PPOTrainer,
            config=conf,
            local_dir=run_base_folder_str,
            #stop={"episodes_total": run_conf.num_episodes}
            stop={"timesteps_total": 10},
            checkpoint_freq=1,
            checkpoint_at_end=True
        )
        best_checkpoint = result.get_last_checkpoint(
            metric="episode_reward_mean", mode="max"
        )
        #best_checkpoint = result.get_trial_checkpoints_paths(
        #    trial=result.get_best_trial("episode_reward_mean"),
        #    metric="episode_reward_mean", mode="max")

        print(best_checkpoint)
        trainer = ppo.PPOTrainer(config=conf)
        trainer.restore(best_checkpoint)
        model = trainer.get_policy().model
        print(type(model))
        print(model)
    else:
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
