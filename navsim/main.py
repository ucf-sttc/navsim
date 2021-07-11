from pathlib import Path
from typing import Optional, List, Set

import attr
import navsim
from .util import ObjDict
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
        "log_folder": str((run_base_folder / "env_log").resolve()),
        "env_path": args["env_path"],
        "worker_id": 0,
        "base_port": 5005,
        "seed": int(args["seed"]),
        "timeout": int(args["timeout"]),
        "obs_mode": int(args["obs_mode"]),
        "obs_height": int(args["obs_height"]),
        "obs_width": int(args["obs_width"]),
        "segmentation_mode": int(args["segmentation_mode"]),
        "episode_max_steps": int(args["episode_max_steps"]),
        "task": int(args["task"]),
        "goal": int(args["goal"]),
        "goal_distance": int(args["goal_distance"]),
        "agent_car_physics": int(args["agent_car_physics"]),
        "reward_for_goal": float(args["reward_for_goal"]),
        "reward_for_no_viable_path": float(args["reward_for_no_viable_path"]),
        "reward_step_mul": float(args["reward_step_mul"]),
        "reward_collision_mul": float(args["reward_collision_mul"]),
        "reward_spl_delta_mul": float(args["reward_spl_delta_mul"]),
        "env_gpu_id": int(args["env_gpu_id"]),
        "debug": args["debug"],
        "save_vector_obs": args["save_vector_obs"],
        "save_visual_obs": args["save_visual_obs"]
    })

    run_conf = ObjDict({
        "run_id": args["run_id"],
        "env":'navsim-v0',
        "agent_gpu_id": int(args["agent_gpu_id"]),
        "num_workers" : 1,
        "episode_max_steps": int(args["episode_max_steps"]),
        "total_episodes": int(args["total_episodes"]),
        "seed": int(args["seed"]),
        "discount": float(args["discount"]),
        "tau": float(args["tau"]),
        "expl_noise": float(args["exploration_noise"]),
        "memory_capacity": int(args["memory_capacity"]),
        "batch_size": int(args["batch_size"]),
        "checkpoint_interval": int(args["checkpoint_interval"]),
        "train_interval": int(args["train_interval"]),
        "mem_backend": args["mem_backend"]
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
        # if resume is passed then read the args from saved conf instead and
        # then overwrite with the args passed
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
                        "checkpoint_interval",
                        "train_interval", "agent_gpu_id"]:
                conf["run_config"][arg] = int(conf["run_config"][arg])
            for arg in ["discount", "tau", "expl_noise"]:
                conf["run_config"][arg] = float(conf["run_config"][arg])
            for arg in ["env_gpu_id", "seed", "timeout", "base_port",
                        "obs_mode", "obs_height", "obs_width",
                        "segmentation_mode", "episode_max_steps", "task",
                        "goal", "goal_distance", "agent_car_physics"]:
                conf["env_config"][arg] = int(conf["env_config"][arg])
            for arg in ["reward_for_goal",
                        "reward_for_no_viable_path",
                        "reward_step_mul", "reward_collision_mul",
                        "reward_spl_delta_mul"]:
                conf["env_config"][arg] = float(conf["env_config"][arg])

        executor = navsim.executor.Executor(run_id=args["run_id"],
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
