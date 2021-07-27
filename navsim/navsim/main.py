from pathlib import Path
from typing import Optional, List, Set

import navsim
from .util.dict import ObjDict
from .cli_utils import argparser, non_default_args
from navsim.executor.navsim_executor import Executor


def main():
    """
    TODO: Implement configuration checks
    :return:
    """
    print(f'=========================================')
    print(f'Navsim Python API Version {navsim.__version__}')
    print(f'========================================')
    args = ObjDict(vars(argparser.parse_args()))
    print('arguments passed:')
    print(non_default_args)
    print("Passed + default arguments:")
    print(args.to_yaml())

    # lets get the arguments
    run_base_folder = Path(args.run_id).resolve()

    env_conf = ObjDict({
        "agent_car_physics": int(args["agent_car_physics"]),
        "debug": args["debug"],
        "episode_max_steps": int(args["episode_max_steps"]),
        "env_gpu_id": int(args["env_gpu_id"]),
        "env_path": args["env_path"],
        "goal": int(args["goal"]),
        "goal_distance": int(args["goal_distance"]),
        "log_folder": str(run_base_folder / "env_log"),
        "obs_mode": int(args["obs_mode"]),
        "obs_height": int(args["obs_height"]),
        "obs_width": int(args["obs_width"]),
        "reward_for_goal": float(args["reward_for_goal"]),
        "reward_for_no_viable_path": float(args["reward_for_no_viable_path"]),
        "reward_step_mul": float(args["reward_step_mul"]),
        "reward_collision_mul": float(args["reward_collision_mul"]),
        "reward_spl_delta_mul": float(args["reward_spl_delta_mul"]),
        "save_actions": args["save_actions"],
        "save_vector_obs": args["save_vector_obs"],
        "save_visual_obs": args["save_visual_obs"],
        "seed": int(args["seed"]),
        "segmentation_mode": int(args["segmentation_mode"]),
        "show_visual": args["show_visual"],
        "task": int(args["task"]),
        "timeout": int(args["timeout"]),
        "traffic_vehicles": int(args["traffic_vehicles"]),
        "worker_id": 0,
        "base_port": 5005,
    })

    run_conf = ObjDict({
        "run_id": args["run_id"],
        "env": 'navsim-v0',
        "agent_gpu_id": int(args["agent_gpu_id"]),
        "num_workers": 1,
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
        "mem_backend": args["mem_backend"],
        "clear_memory": args["clear_memory"],
        "log_level": "DEBUG" if args.debug else "INFO",
        "framework": "torch"
    })

    if args.resume:
        if run_base_folder.exists():
            # TODO: Logic for resume here
            pass

        else:  # else just start fresh
            args.resume = False
            print(
                "Resume set to True, but nothing to resume from, starting fresh")

    # start fresh logic
    if not args.resume:
        run_base_folder.mkdir(parents=True, exist_ok=True)



    if args["rl_backend"] == "rllib":
        import ray.rllib.agents.ppo as ppo
        conf = ppo.DEFAULT_CONFIG.copy()
        # copy from run_config to conf
        for key in run_conf:
            if key in conf:
                conf[key] = run_conf[key]
        conf["ignore_worker_failures"] = True
    else:
        conf = ObjDict()
        conf['run_config'] = run_conf
    conf['env_config'] = env_conf

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
            local_dir=str(run_base_folder),
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
            conf = ObjDict().load_from_json_file(
                str(run_base_folder / "conf.json"))

            # detect if physics lebvel has changed in resume
            if ("agent_car_physics" in non_default_args) and int(
                    conf["env_config"]["agent_car_physics"]) != int(
                args["agent_car_physics"]):
                # detect if clear memory was overridden from command prompt
                if "clear_memory" not in non_default_args:
                    conf["run_config"]["clear_memory"] = True
                else:
                    # clear_memory will be set from passed args below
                    pass

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
                        "goal", "goal_distance", "traffic_vehicles",
                        "agent_car_physics"]:
                conf["env_config"][arg] = int(conf["env_config"][arg])
            for arg in ["reward_for_goal",
                        "reward_for_no_viable_path",
                        "reward_step_mul", "reward_collision_mul",
                        "reward_spl_delta_mul"]:
                conf["env_config"][arg] = float(conf["env_config"][arg])

        executor = Executor(resume=args["resume"],
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
