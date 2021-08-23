from pathlib import Path

import navsim
import navsim_envs

from ezai_util.dict import ObjDict
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
    print(non_default_args if non_default_args else "None")
    print("Passed + default arguments:")
    print(args.to_yaml())

    """
    * read from file / defaults
    1.  If resume or continue:
            Read from config file
        Else
            Read default config (navsim)
            If ray as backend
                Update to Ray's format
        
    If resume or continue
        Read from saved yaml file
    Else
        Get from default configs (ray/navsim-lab-trainer)
    * Override with CLI
        IF resume
            No override from CLI as its just resuming a crashed or ^c’ed run
        ElIf continue and cli_physics_level changed
            If cli_clear_memory not specified
                Clear_memory = True
            Else nothing
        Else nothing
        If continue or new training:
            Set values from cli_*
        else
            nothing
    * Update respective values as float / int
    """
    if args.resume and args.continue_arg:
        raise ValueError("Both Resume and continue passed from CLI, please pass only one")

    if args.continue_arg and args.rl_backend == "rllib":
        raise ValueError("RLLib doesnt support continue")

    if args.resume and args.rl_backend == "navsim":
        raise ValueError("navsim doesnt support resume")

    # lets get the arguments
    run_base_folder = Path(args.run_id).resolve()

    run_config = None
    env_config = None

    if args.resume or args.continue_arg:
        if run_base_folder.exists():
            run_config = ObjDict.load_from_file(str(run_base_folder / "run_config.yml"))
            env_config = ObjDict.load_from_file(str(run_base_folder / "env_config.yml"))
        else:
            args.resume = False
            args.continue_arg = False
            print("Resume or continue passed from CLI, but nothing to resume/continue from, starting fresh")
            # start fresh logic comes next

    # start fresh logic is separate below because
    # resume may be set to false in nested else above
    if not (args.resume or args.continue_arg):
        import shutil
        if run_base_folder.exists():
            shutil.rmtree(run_base_folder)
        # run_config = ObjDict()
        # TODO: run_config = navsim.default_conf
        run_config = ObjDict(navsim.util.run_config.copy())
        env_config = ObjDict(navsim_envs.arora.default_env_config.copy())
        env_log_folder = run_base_folder / 'env_log'
        env_log_folder.mkdir(parents=True, exist_ok=True)
        env_config["log_folder"] = str(env_log_folder)

    # TODO: can we also implement it for non-navsim backends ?
    if args.rl_backend == "navsim":
        # in case of continue, lets check if memory needs to be invalidated
        if args.continue_arg:
            # detect if physics level has changed in continue_arg
            if ("agent_car_physics" in non_default_args) and (
                    int(env_config["agent_car_physics"]) != int(args["agent_car_physics"])):
                # detect if clear memory was overridden from command prompt
                if "clear_memory" not in non_default_args:
                    run_config["clear_memory"] = True
                else:
                    # clear_memory will be set from passed args below
                    pass
            else:
                pass
        else:
            pass

    # in case of continue and fresh start, lets set configs with CLI
    if not args.resume:
        run_config.log_level = "DEBUG" if args.debug else "INFO"
        for passed_arg in non_default_args:
            if passed_arg in env_config:
                env_config[passed_arg] = args[passed_arg]
            if passed_arg in run_config:
                run_config[passed_arg] = args[passed_arg]

    # now lets sanitize the conf and set appropriate ints and floats
    int_args = ["agent_gpu_id", "batch_size", "checkpoint_interval", "episode_max_steps", "memory_capacity", "seed",
                "total_episodes", "train_interval"] + [
                   "agent_car_physics", "base_port", "episode_max_steps", "env_gpu_id", "goal", "goal_distance",
                   "obs_mode",
                   "obs_height", "obs_width", "segmentation_mode", "task", "timeout", "traffic_vehicles"]
    float_args = ["discount", "tau", "expl_noise"] + ["reward_for_goal", "reward_for_no_viable_path", "reward_step_mul",
                                                      "reward_collision_mul", "reward_spl_delta_mul"]
    for arg in int_args:
        if arg in run_config:
            run_config[arg] = None if run_config[arg] is None else int(run_config[arg])
        if arg in env_config:
            env_config[arg] = None if env_config[arg] is None else int(env_config[arg])

    for arg in float_args:
        if arg in run_config:
            run_config[arg] = None if run_config[arg] is None else float(run_config[arg])
        if arg in env_config:
            env_config[arg] = None if env_config[arg] is None else float(env_config[arg])

    print("Final Run Configuration:")
    print(run_config.to_yaml())
    print("Final Env Configuration:")
    print(env_config.to_yaml())
    run_base_folder.mkdir(parents=True, exist_ok=True)
    run_config.save_to_yaml_file(str(run_base_folder / "run_config.yml"))
    env_config.save_to_yaml_file(str(run_base_folder / "env_config.yml"))

    """
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
    """

    if args["rl_backend"] == "rllib":
        import ray.rllib.agents.ppo as ppo
        config = ObjDict(ppo.DEFAULT_CONFIG.copy())
        for arg in run_config:
            if arg in config:
                config[arg] = run_config[arg]
        config["env_config"] = env_config
        config["ignore_worker_failures"] = True
        # TODO: Override ray's conf with some defaults from navsim
        import ray
        ray.shutdown()
        ray.init(ignore_reinit_error=True)
        navsim_envs.env.AroraGymEnv.register_with_ray()
        result = ray.tune.run(
            ppo.PPOTrainer,
            config=config,
            name=run_config.run_id,
            resume=run_config.resume,
            local_dir=str(run_base_folder),
            stop={"episodes_total": run_config.total_episodes},
            checkpoint_freq=run_config.checkpoint_interval,
            checkpoint_at_end=True
        )
        best_checkpoint = result.get_last_checkpoint(
            metric="episode_reward_mean", mode="max"
        )
        # best_checkpoint = result.get_trial_checkpoints_paths(
        #    trial=result.get_best_trial("episode_reward_mean"),
        #    metric="episode_reward_mean", mode="max")

        print(best_checkpoint)
        trainer = ppo.PPOTrainer(config=config)
        trainer.restore(best_checkpoint)
        model = trainer.get_policy().model
        print(type(model))
        print(model)
    else:
        config = run_config
        config["env_config"] = env_config

        executor = Executor(config=config)
        executor.execute()
        print("training finished")

        executor.env_close()
        print("env closed")
        executor.files_close()
        print("files closed")


# For python debugger to directly run this script
if __name__ == "__main__":
    main()
