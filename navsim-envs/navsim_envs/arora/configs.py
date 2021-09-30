import yaml

default_env_config = {
    "agent_car_physics": 0,
    "debug": False,
    "episode_max_steps": 1000,
    "env_gpu_id": 0,
    "env_path": None,
    "goal": 0,
    "goal_distance": 50,
    "goal_clearance": 2.5,
    "log_folder": "./env_log",
    "obs_mode": 0,
    "obs_height": 64,
    "obs_width": 64,
    "seed": None,
    "segmentation_mode": 1,
    "start_from_episode": 1,
    "reward_for_goal": 50,
    "reward_for_no_viable_path": -50,
    "reward_step_mul": 0.1,
    "reward_collision_mul": 4,
    "reward_spl_delta_mul": 1,
    "relative_steering": True,
    "save_actions": True,
    "save_vector_obs": True,
    "save_visual_obs": True,
    "show_visual": False,
    "task": 0,
    "timeout": 600,
    "traffic_vehicles": 0,
    "worker_id": 0,
    "base_port": 5004,
}
