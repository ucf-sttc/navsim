from pathlib import Path

import cv2
import gym
import navsim
import navsim_envs

from ezai.util import ObjDict
from matplotlib import pyplot as plt
from navsim.planner.navsim_planner import NavsimPlanner

from .cli_utils import argparser, non_default_args
from navsim.executor.navsim_executor import Executor

def increase_brightness(image, value=0.1):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv)
    # lim = 255 - value
    # v[v > lim] = 255
    # v[v <= lim] += value
    # final_hsv = cv2.merge((h, s, v))

    hsv[:,:,2] = cv2.add(hsv[:, :, 2], value )
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image

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
        run_config = ObjDict(navsim.util.run_config.copy())
        env_config = ObjDict(navsim.util.env_config.copy())
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
    float_args = ["discount", "tau", "expl_noise"] + ["goal_clearance", "reward_for_goal", "reward_for_no_viable_path",
                                                      "reward_step_mul",
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

    if args["plan"] is True:
        run_config["total_episodes"]=2
        #env_config["episode_max_steps"]= 10000
        #env_config["goal_clearance"] = 20
        #env_config["goal_distance"]= 100
        env_config["obs_mode"] = 1
        #env_config["obs_height"] = 256
        #env_config["obs_width"] = 256
        #env_config["seed"] = 12345
        env_config["relative_steering"] = False
        env = gym.make(run_config["env"], env_config=env_config)
        for episode_num in range(0,run_config["total_episodes"]):
            o = env.reset()
            planner = NavsimPlanner(env)
            num_step = 0
            a = False
            d = False
            plt.ion()
            while (a is not None) and (d is False):
                a = planner.plan(o)
                if a is None:
                    break
                o, r, d, i = env.step(a)
                print("distance:", env.shortest_path_length, " | reward:", r)
    #            if d:
    #                break
                num_step += 1
                # replaced with --show_visual from unity
                #if num_step % 3 == 0:
                #    #print(type(env))
                #    plt.imshow(increase_brightness(env.render(mode='rgb_array')))
                #    plt.show()
                #    plt.pause(0.001)
                #    plt.clf()
    elif args["rl_backend"] == "rllib":
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
        navsim_envs.env.RideGymEnv.register_with_ray()
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
