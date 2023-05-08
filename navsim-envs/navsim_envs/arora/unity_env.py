#ARORA

import time
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from .configs import default_env_config
from navsim_envs.envs_base import AroraUnityEnvBase

class AroraUnityEnv(AroraUnityEnvBase):
    """AroraUnityEnv Class is a wrapper to UnityEnvironment

    Read the **NavSim Environment Tutorial** on how to use this class.
    """

    actions = {
        'forward_left' : [1,-1,0],
        'forward_right' :[1, 1,0],
        'forward' : [1,0,0],
        'backward' : [-1,0,0]
    }

    def __init__(self, env_config) -> None:
        """
        env_config: The environment configuration dictionary Object
        """

        super().__init__(env_config,default_env_config)
        del env_config

        eng_sc = EngineConfigurationChannel()
        eng_sc.set_configuration_parameters(time_scale=0.25, quality_level=0)
        env_pc = EnvironmentParametersChannel()

        timeout = self.env_config['timeout'] + (0.5 * (self.env_config['start_from_episode'] - 1))

        ad_args = [
            "-allowedArea", f"{self.env_config['area']}",
            "-agentCarPhysics", f"{self.env_config['agent_car_physics']}",
            "-episodeLength", f"{self.env_config['episode_max_steps']}",
            "-fastForward", f"{self.env_config['start_from_episode'] - 1}",
            "-force-device-index", f"{self.env_config['env_gpu_id']}",
            "-force-vulkan" if (self.env_config["env_gpu_id"] > 0) else "",
            "-goalDistance", f"{self.env_config['goal_distance']}",
            "-goalClearance", f"{self.env_config['goal_clearance']}",
            "-goalSelectionIndex", f"{self.env_config['goal']}",
            "-numberOfTrafficVehicles", f"{self.env_config['traffic_vehicles']}",
            "-observationMode", f"{self.env_config['obs_mode']}",
            "-observationWidth", f"{self.env_config['obs_width']}",
            "-observationHeight", f"{self.env_config['obs_height']}",
            "-relativeSteering", f"1" if self.env_config['relative_steering'] else f"0",
            "-rewardForGoal", f"{self.env_config['reward_for_goal']}",
            "-rewardForNoViablePath", f"{self.env_config['reward_for_no_viable_path']}",
            "-rewardStepMul", f"{self.env_config['reward_step_mul']}",
            "-rewardCollisionMul", f"{self.env_config['reward_collision_mul']}",
            "-rewardSplDeltaMul", f"{self.env_config['reward_spl_delta_mul']}",
            "-showVisualObservations" if self.env_config['show_visual'] else "",
            "-saveStepLog" if self.env_config["debug"] else "",
            "-segmentationMode", f"{self.env_config['segmentation_mode']}",
            "-selectedTaskIndex", f"{self.env_config['task']}"
        ]

        self._navsim_base_port = self.env_config['base_port']
        if self._navsim_base_port is None:
            self._navsim_base_port = UnityEnvironment.BASE_ENVIRONMENT_PORT if env_config[
                'env_path'] else UnityEnvironment.DEFAULT_EDITOR_PORT
        self._navsim_worker_id = self.env_config['worker_id']

        while True:
            try:
                self.env_config["worker_id"] = self._navsim_worker_id
                self.env_config["base_port"] = self._navsim_base_port
                UnityEnvironment.__init__(self,file_name=self.env_config['env_path'],
                                 log_folder=self.env_config["log_folder"],
                                 no_graphics=False,
                                 seed=self.env_config["seed"],
                                 timeout_wait=timeout,
                                 worker_id=self.env_config["worker_id"],
                                 base_port=self.env_config["base_port"],
                                 side_channels=[eng_sc, env_pc,
                                                self.map_side_channel,
                                                self.fpc, self.nsc,
                                                self.sapsc, self.spsc
                                                ],
                                 additional_args=ad_args)
                
            except UnityWorkerInUseException:
                time.sleep(2)
                self._navsim_base_port += 1
            else:
                from_str = "Editor" if self.env_config['env_path'] is None else f"from {self.env_config['env_path']}"
                AroraUnityEnv.logger.info(f"Created UnityEnvironment {from_str} "
                                         f"at port {self._navsim_base_port + self._navsim_worker_id} "
                                         f"to start from episode {self.env_config['start_from_episode']}")
                break

        #self.env_config = env_config




    

