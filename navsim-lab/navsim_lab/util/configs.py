run_config = {
    "run_id": 'demo',
    "env": 'arora-v0',
    "agent_gpu_id": 0,
    "num_workers": 1,
    "episode_max_steps": 50,
    "total_episodes": 1,
    "seed": None,
    "discount": 0.99,
    "tau": 5e-3,
    "expl_noise": 0.1,
    "memory_capacity": 100,
    "batch_size": 32,
    "checkpoint_interval": 1,
    "train_interval": 1,
    "mem_backend": "cupy",
    "clear_memory": False,
    "debug": False,
    "log_level": "INFO",
    "framework": "torch",
    "resume": False,
    "continue_arg": False,
    "plan": False
}

env_config = {
    "agent_car_physics": 0,
    "area": 0,
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
    "terrain": 0,
    "timeout": 600,
    "traffic_vehicles": 0,
    "worker_id": 0,
    "base_port": 5004,
}
