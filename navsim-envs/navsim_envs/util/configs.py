import yaml

env_config = {
    "agent_car_physics": 0,
    "debug": False,
    "episode_max_steps": 1000,
    "env_gpu_id": 0,
    "env_path": None,
    "goal": 0,
    "goal_distance": 50,
    "log_folder": "./env_log",
    "obs_mode": 0,
    "obs_height": 256,
    "obs_width": 256,
    "seed": 123,
    "start_from_episode": 1,
    "reward_for_goal": 50,
    "reward_for_no_viable_path": -50,
    "reward_step_mul": 0.1,
    "reward_collision_mul": 4,
    "reward_spl_delta_mul": 1,
    "save_actions": True,
    "save_vector_obs": True,
    "save_visual_obs": True,
    "show_visual": False,
    "task": 0,
    "timeout": 600,
    "traffic_vehicles": 0,
    "worker_id": 0,
    "base_port": 5005,
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
