from pathlib import Path

from mlagents_envs.environment import UnityEnvironment

from mlagents_envs.logging_util import get_logger
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
from navsim_envs.env.map_side_channel import MapSideChannel
from navsim_envs.env.navigable_side_channel import NavigableSideChannel
from navsim_envs.env.set_agent_position_side_channel import SetAgentPositionSideChannel

from ..util.configs import env_config as default_env_config


class AroraUnityEnv(UnityEnvironment):
    """AroraGymEnv Class is a wrapper to Unity2Gym that inherits from the Gym interface

    Read the **NavSim Environment Tutorial** on how to use this class.
    """
    logger = get_logger(__name__)

    def __init__(self, env_config) -> None:
        """
        env_config: The environment configuration dictionary Object
        """

        for key in default_env_config:
            if key not in env_config:
                env_config[key] = env_config

        log_folder = Path(env_config['log_folder']).resolve()
        log_folder.mkdir(parents=True, exist_ok=True)

        self.map_side_channel = MapSideChannel()
        self.fpc = FloatPropertiesChannel()
        self.nsc = NavigableSideChannel()
        self.sapsc = SetAgentPositionSideChannel()

        eng_sc = EngineConfigurationChannel()
        eng_sc.set_configuration_parameters(time_scale=10, quality_level=0)

        env_pc = EnvironmentParametersChannel()
        env_sfp = env_pc.set_float_parameter

        env_sfp("rewardForGoal", env_config['reward_for_goal'])
        env_sfp("rewardForNoViablePath", env_config['reward_for_no_viable_path'])
        env_sfp("rewardStepMul", env_config['reward_step_mul'])
        env_sfp("rewardCollisionMul", env_config['reward_collision_mul'])
        env_sfp("rewardSplDeltaMul", env_config['reward_spl_delta_mul'])
        env_sfp("segmentationMode", env_config['segmentation_mode'])
        env_sfp("observationMode", env_config['obs_mode'])
        env_sfp("episodeLength", env_config['episode_max_steps'])
        env_sfp("selectedTaskIndex", env_config['task'])
        env_sfp("goalSelectionIndex", env_config['goal'])
        env_sfp("agentCarPhysics", env_config['agent_car_physics'])
        env_sfp("goalDistance", env_config['goal_distance'])
        env_sfp("numberOfTrafficVehicles", env_config['traffic_vehicles'])

        timeout = env_config['timeout'] + (0.5 * (env_config['start_from_episode'] - 1))

        ad_args = [
            # "-force-device-index",
            "-gpu",
            f"{env_config['env_gpu_id']}",
            "-observationWidth",
            f"{env_config['obs_width']}",
            "-observationHeight",
            f"{env_config['obs_height']}",
            "-fastForward", f"{env_config['start_from_episode'] - 1}",
            "-showVisualObservations" if env_config['show_visual'] else "",
            "-saveStepLog" if env_config["debug"] else ""
        ]

        super().__init__(file_name=env_config['env_path'],
                         log_folder=env_config["log_folder"],
                         no_graphics=False,
                         seed=env_config["seed"],
                         timeout_wait=timeout,
                         worker_id=env_config["worker_id"],
                         base_port=env_config["base_port"],
                         side_channels=[eng_sc, env_pc,
                                        self.map_side_channel,
                                        self.fpc, self.nsc,
                                        self.sapsc],
                         additional_args=ad_args)

        self.env_config = env_config
