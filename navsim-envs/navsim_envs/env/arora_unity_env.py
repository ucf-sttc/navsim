from mlagents_envs.environment import UnityEnvironment


class NavSimUnityEnv(UnityEnvironment):
    """AroraGymEnv Class is a wrapper to Unity2Gym that inherits from the Gym interface

    Read the **NavSim Environment Tutorial** on how to use this class.
    """

    def __init__(self, env_config) -> None:
        """
        env_config: The environment configuration dictionary Object
        """

        env_path = env_config.get('env_path', None)
        log_folder = str(env_config.get('log_folder', './env_log'))
        seed = env_config.get('seed', 0)
        timeout = self.env_config.get('timeout', 600) + (0.5 * (
                self.start_from_episode - 1))
        worker_id = env_config.get('worker_id', 0)
        base_port = env_config.get('base_port', None)

        super().__init__(file_name=env_path,
                         log_folder=str(log_folder),
                         seed=seed,
                         timeout_wait=timeout,
                         worker_id=worker_id,
                         base_port=base_port,
                         no_graphics=False,
                         side_channels=[],
                         additional_args="")


