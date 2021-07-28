from pathlib import Path

import yaml

run_base_folder = Path('.').resolve() / 'tst_logs'
run_base_folder.mkdir(parents=True, exist_ok=True)

env_config = {
    "log_folder": str(run_base_folder / "env_log"),
    "env_path": "/data/work/unity-envs/Build2.10.2/Berlin_Walk_V2.x86_64",
    "worker_id": 0,
    "base_port": 5005,
    "seed": 1,
    "timeout": 120,
    "obs_mode": 0,
    "obs_height": 128,
    "obs_width": 128,
    "segmentation_mode": 0,
    "episode_max_steps": 1000,
    "task": 0,
    "goal": 0,
    "goal_distance": 10,
    "traffic_vehicles": 0,
    "agent_car_physics": 0,
    "reward_for_goal": 1.0,
    "reward_for_no_viable_path": -1.0,
    "reward_step_mul": 1.0,
    "reward_collision_mul": 1.0,
    "reward_spl_delta_mul": 1.0,
    "env_gpu_id": 0,
    "debug": False,
    "save_actions": False,
    "save_vector_obs": False,
    "save_visual_obs": False,
    "show_visual": False
}


def config_banner(config, config_name):
    return f'----------------------------------------\n' \
           f'{config_name}\n' \
           f'----------------------------------------\n' \
           f'{yaml.dump(config, default_flow_style=False)}' \
           f'----------------------------------------\n'


def save_config(config, config_filename):
    with open(config_filename, 'w') as f:
        val = yaml.dump(config, f, default_flow_style=False)
    return val


def load_config(config_filename):
    with open(config_filename, 'r') as f:
        return yaml.safe_load(f)
