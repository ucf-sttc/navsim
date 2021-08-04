from pathlib import Path
import random

import gym
import sys
import numpy as np
import pytest
from copy import deepcopy

import navsim_envs
from navsim_envs.env import AroraGymEnv
from navsim_envs.util.configs import config_banner, load_config
from navsim_envs.util.configs import env_config as env_conf

from scipy.spatial.transform import Rotation as R

logger = AroraGymEnv.logger

#def setLoggerLevel(debug):
#    if debug:
#        logger.setLevel(10)
#    else:
#        logger.setLevel(20)

# This runs first before any class in this module runs
@pytest.fixture(scope="module")
def env_config():
    config = deepcopy(env_conf)
    run_base_folder = Path('.').resolve() / 'tst_logs'
    run_base_folder.mkdir(parents=True, exist_ok=True)
    config["log_folder"] = str(run_base_folder / "env_log")
    if len(sys.argv) > 1:
        env_config_file = sys.argv.pop()
        env_config_from_file = load_config(env_config_file)
        if env_config_from_file is not None:
            logger.info(f'Updating env_config from {env_config_file}')
            config.update(env_config_from_file)

    logger.info(f'\n'
                f'{config_banner(config, "env_config")}'
                )
    return config


def env_deletor(env):
    env.close()
    del env


@pytest.fixture(scope="class")
def env_4_class():
    env = None

    def env_creator(env_config):
        nonlocal env
        if env is None:
            env = gym.make("arora-v0", env_config=env_config)
        return env

    yield env_creator
    env_deletor(env)


@pytest.fixture(scope="function")
def env_4_func():
    env = None

    def env_creator(env_config):
        nonlocal env
        if env is None:
            env = gym.make("arora-v0", env_config=env_config)
        return env

    yield env_creator
    env_deletor(env)

VALID_QUATERNIONS = [[0, 0.985997, 0, -0.1667631], [0, 0.7649251, 0, -0.6441193], [0, 0.6004248, 0, -0.7996812], [0, 0.2486954, 0, -0.9685818], [0, -0.08019328, 0, -0.9967793], [0, -0.9727746, 0, -0.2317534], [0, -0.08542251, 0, 0.9963449], [0, 0.9488743, 0, 0.3156544], [0, -0.7501073, 0, -0.6613162], [0, 0.463291, 0, 0.8862062], [0, 0.873775, 0, -0.4863304], [0, -0.9812938, 0, 0.1925164], [0, -0.08542265, 0, 0.9963449], [0, 0.3354462, 0, 0.9420595], [0, 0.3518363, 0, 0.9360616], [0, 0.579286, 0, -0.8151245], [0, -0.9989709, 0, 0.0453571], [0, -0.9449509, 0, 0.3272124], [0, -0.006987173, 0, 0.9999756], [0, 0.9887573, 0, -0.1495295], [0, -0.335446, 0, -0.9420595], [0, -0.9916704, 0, -0.1288015], [0, 0.7836898, 0, 0.6211524], [0, -0.6032034, 0, -0.7975875], [0, -0.747802, 0, 0.6639219], [0, 0.7325389, 0, 0.6807252], [0, 0.9905087, 0, 0.1374503], [0, 0.5431793, 0, -0.8396167], [0, 0.5284433, 0, -0.8489686], [0, -0.9342024, 0, -0.3567434], [0, -0.2739648, 0, 0.9617398], [0, 0.3599914, 0, 0.9329557], [0, 0.9033328, 0, 0.4289404], [0, 0.03316105, 0, -0.9994501], [0, -0.9018351, 0, 0.4320805], [0, 0.8151244, 0, 0.579286], [0, 0.9874148, 0, -0.1581522], [0, -0.6238749, 0, -0.7815243], [0, -0.9905087, 0, -0.1374506], [0, -0.7975876, 0, 0.6032031], [0, 0.9568118, 0, 0.2907081], [0, 0.7975877, 0, -0.6032031], [0, 0.3891297, 0, -0.921183], [0, -0.926531, 0, 0.3762185], [0, -0.2402341, 0, 0.970715], [0, -0.006987521, 0, 0.9999756], [0, 0.235136, 0, 0.9719625], [0, 0.9033327, 0, 0.4289409], [0, -0.8796456, 0, -0.4756299], [0, 0.767161, 0, 0.6414545], [0, -0.6959082, 0, -0.7181308], [0, 0.6639213, 0, 0.7478024], [0, 0.9967794, 0, -0.08019239], [0, 0.8231401, 0, -0.5678384], [0, 0.699668, 0, -0.7144681], [0, -0.5962197, 0, -0.8028214], [0, -0.9851083, 0, -0.1719356], [0, -0.9128368, 0, 0.4083244], [0, 0.03663707, 0, 0.9993286], [0, 0.07149082, 0, 0.9974413], [0, 0.5284439, 0, -0.8489683], [0, -0.1667623, 0, -0.9859972], [0, -0.8625103, 0, -0.5060394], [0, 0.695908, 0, 0.7181309], [0, 0.3239239, 0, -0.9460832], [0, -0.9449512, 0, 0.3272114], [0, -0.8424561, 0, 0.538765], [0, 0.8821301, 0, -0.4710059], [0, -0.9685815, 0, -0.2486965], [0, 0.447753, 0, 0.8941573], [0, -0.4162746, 0, -0.9092389], [0, -0.5934244, 0, 0.8048897], [0, -0.03316204, 0, 0.99945], [0, 0.9420598, 0, -0.3354451], [0, -0.7144678, 0, -0.6996683], [0, -0.9515923, 0, -0.3073632], [0, 0.5387649, 0, 0.8424562], [0, 0.5962193, 0, 0.8028217], [0, 0.1719361, 0, -0.9851081], [0, -0.3106697, 0, -0.950518], [0, -0.9996104, 0, 0.02791462], [0, 0.9432203, 0, 0.3321677], [0, -0.8754612, 0, -0.4832885], [0, 0.4320795, 0, 0.9018356], [0, 0.8878122, 0, 0.460206], [0, 0.3485787, 0, -0.9372795], [0, -0.9845045, 0, 0.1753598], [0, 0.1235945, 0, 0.9923328], [0, 0.986571, 0, 0.1633329], [0, 0.9795766, 0, -0.201071], [0, -0.5165273, 0, -0.8562707], [0, -0.8377226, 0, 0.546096], [0, -0.4289416, 0, 0.9033323], [0, 0.5284443, 0, -0.8489679], [0, -0.4863292, 0, -0.8737757], [0, -0.08542409, 0, 0.9963447], [0, 0.7558488, 0, 0.6547463], [0, 0.4679362, 0, -0.8837622], [0, -0.4555385, 0, -0.8902162], [0, -0.5284446, 0, 0.8489679]]

"""
TestNavSimGymEnv1 : env once per class
TestNavSimGymEnv2 : env once per test
TestNavSimGymEnv3 : env more than once within a test
"""


class TestAroraGymEnv1:
    """
    env is initialized once per class
    """

    # Tests whether the navigable map returned has navigable points
    def test_navigable_map(self, request, env_4_class, env_config):
        env = env_4_class(env_config)
        logger.info(f"=========== Running {request.node.name}")
        navigable_map = env.get_navigable_map()
        logger.debug(type(navigable_map))
        logger.debug(navigable_map)
        logger.info(f'Navigable cells: {navigable_map.sum()}')
        assert navigable_map.sum() > 1

    # Tests whether the navmap to unity and back coordinate conversion are correct
    def test_coordinate_conversion(self, request, env_4_class, env_config):
        env = env_4_class(env_config)
        logger.info(f"=========== Running {request.node.name}")
        navmap_max_x = 256
        navmap_max_y = 256

        resolution_margin = 15

        unity_loc_original = (0, 0)

        navmap_loc = env.unity_to_navmap_location(*unity_loc_original,
                                                 navmap_max_x=navmap_max_x,
                                                 navmap_max_y=navmap_max_y)
        unity_loc = env.navmap_to_unity_location(*navmap_loc,
                                                navmap_max_x=navmap_max_x,
                                                navmap_max_y=navmap_max_y,
                                                navmap_cell_center=False)
        logger.info("unity to nav to unity")
        logger.info(f'{unity_loc_original}, {navmap_loc}, {unity_loc}')
        assert navmap_loc == (0, 0)
        assert unity_loc == (0, 0)

        unity_loc_original = (0, 2665)
        navmap_loc = env.unity_to_navmap_location(*unity_loc_original,
                                                 navmap_max_x=navmap_max_x,
                                                 navmap_max_y=navmap_max_y)
        unity_loc = env.navmap_to_unity_location(*navmap_loc,
                                                navmap_max_x=navmap_max_x,
                                                navmap_max_y=navmap_max_y,
                                                navmap_cell_center=False)
        logger.info(f'{unity_loc_original}, {navmap_loc}, {unity_loc}')
        assert navmap_loc == (0, 255)
        assert unity_loc[0] == 0 and (
                unity_loc[1] > 2665 - resolution_margin and unity_loc[
            1] < 2665 + resolution_margin)

        unity_loc_original = (3283, 0)
        navmap_loc = env.unity_to_navmap_location(*unity_loc_original,
                                                 navmap_max_x=navmap_max_x,
                                                 navmap_max_y=navmap_max_y)
        unity_loc = env.navmap_to_unity_location(*navmap_loc,
                                                navmap_max_x=navmap_max_x,
                                                navmap_max_y=navmap_max_y,
                                                navmap_cell_center=False)
        logger.info(f'{unity_loc_original}, {navmap_loc}, {unity_loc}')
        assert navmap_loc == (255, 0)
        assert (unity_loc[0] > 3283 - resolution_margin and unity_loc[
            0] < 3283 + resolution_margin) and unity_loc[1] == 0

        unity_loc_original = (3283, 2665)
        navmap_loc = env.unity_to_navmap_location(*unity_loc_original,
                                                 navmap_max_x=navmap_max_x,
                                                 navmap_max_y=navmap_max_y)
        unity_loc = env.navmap_to_unity_location(*navmap_loc,
                                                navmap_max_x=navmap_max_x,
                                                navmap_max_y=navmap_max_y,
                                                navmap_cell_center=False)
        logger.info(f'{unity_loc_original}, {navmap_loc}, {unity_loc}')
        assert navmap_loc == (255, 255)
        assert (unity_loc[0] > 3283 - resolution_margin and unity_loc[
            0] < 3283 + resolution_margin) and (
                       unity_loc[1] > 2665 - resolution_margin and
                       unity_loc[1] < 2665 + resolution_margin)


    # Tests sample navigable point and setting the agent's position
    def test_set_agent_position(self, request, env_4_class, env_config):
        logger.info(f"=========== Running {request.node.name}")
        #setLoggerLevel(env_config["debug"])
        
        env = env_4_class(env_config)
        navigable_map = env.get_navigable_map()
        env.reset()
        err_margin=1.0
        samples = 1000
        for i in range(0,samples):
            logger.debug(f"===========")
            logger.debug(f"Original Position: {env.agent_position} and {env.agent_rotation} and {env.goal_position}")
            sampled_position = env.sample_navigable_point() 
            success = env.set_agent_position(sampled_position)
            logger.debug(f"Sampled Position: {sampled_position}")
            o, r, done, i = env.step([0, 0, 1])

            #TODO Replace with env.agent_position and env.agent_rotation after they are updated
            #logger.info(f"After Agent Set: {env.agent_position} and {env.agent_rotation} and {env.goal_position}")
            cur_position = o[0][:3]
            cur_rotation = o[0][6:10]
            logger.debug(f"Observation: {o}")
            logger.debug(f"Position {cur_position} and Rotation {cur_rotation}")
            
            logger.debug(f"Set Agent Position Request Returned : {success}")
            assert success == True
            assert cur_position[0] < sampled_position[0]+err_margin and cur_position[0] > sampled_position[0]-err_margin
            assert cur_position[1] < sampled_position[1]+err_margin and cur_position[1] > sampled_position[1]-err_margin
            assert cur_position[2] < sampled_position[2]+err_margin and cur_position[2] > sampled_position[2]-err_margin
            
        logger.info(f'{samples} sampled points are able to set agent state')
        
    def test_set_agent_rotation(self, request, env_4_class, env_config):
        logger.info(f"=========== Running {request.node.name}")
        #setLoggerLevel(env_config["debug"])
        
        env = env_4_class(env_config)
        navigable_map = env.get_navigable_map()
        err_margin=0.1
        samples = 1000
        for sample in range(0, samples):
            env.reset()
            for sampled_rotation in VALID_QUATERNIONS:
                logger.debug(f"===========")
                logger.debug(f"Original Position: {env.agent_position} and {env.agent_rotation} and {env.goal_position}")
                sampled_rotation_euler = env.unity_rotation_in_euler(sampled_rotation)

                success = env.set_agent_rotation(sampled_rotation)
                logger.debug(f"Sampled Rotation: {sampled_rotation}")
                o, r, done, i = env.step([0, 0, 1])

                #TODO Replace with env.agent_position and env.agent_rotation after they are updated
                cur_position = o[0][:3]
                cur_rotation = o[0][6:10]
                cur_rotation_euler = env.unity_rotation_in_euler(o[0][6:10])
                logger.debug(f"Observation: {o}")
                logger.debug(f"Position {cur_position} and Rotation {cur_rotation}")

                logger.debug(f"Euler {sampled_rotation_euler} {cur_rotation_euler}")
                
                logger.debug(f"Set Agent Position Request Returned : {success}")
                assert success == True
                assert cur_rotation_euler[1] < sampled_rotation_euler[1]+err_margin and cur_rotation_euler[1] > sampled_rotation_euler[1]-err_margin

            logger.debug(f'reset {sample} completed successfully')
             
            
        logger.info(f'{samples} sampled points are able to set agent state')


class TestAroraGymEnv3:
    """
    env is initialized multiple times in each test
    """

    # Tests whether mnutiple runs produces the same observation values

    def test_reproducibility(self, request, env_config):
        logger.info(f"=========== Running {request.node.name}")
        throttle = 1
        obs_arr = []
        runs = 3
        for outer in range(0, runs):
            env = gym.make("arora-v0", env_config=env_config)
            steering = 0
            obs_inner_arr = []

            o = env.reset()
            logger.debug(o)
            obs_inner_arr.append(o)

            for x in range(0, 10):
                o, r, done, i = env.step([throttle, steering, -1])
                obs_inner_arr.append(o)
                logger.debug(o)

            obs_arr.append(obs_inner_arr)
            env.close()
            del env

        for i in range(1, len(obs_arr)):
            assert np.array_equal(obs_arr[i - 1], obs_arr[i])
            # assert np.isclose(obs_arr[i-1], obs_arr[i], atol=1.0)
        logger.info(
            f'{runs} env runs produced equal observations for the first episode')

    # Tests whether the agent rotates left/right give positive and negative steering
    def test_rotation(self, request, env_config):
        logger.info(f"=========== Running {request.node.name}")
        for steering in [-1, 1]:
            throttle = 1
            env = gym.make("arora-v0", env_config=env_config)

            logger.info(f'Steering is: {steering}')
            rot_arr = []
            o = env.reset()
            logger.debug('Rot Y in euler, Rot Y in quaternion')
            for x in range(0, 10):
                o, r, done, i = env.step([throttle, steering, -1])
                rot = o[0][7]
                logger.debug(f'{env.unity_rotation_in_euler()[2]}, {rot}')
                rot_arr.append((env.unity_rotation_in_euler()[2], rot))

            env.close()
            del env

            for i in range(1, len(rot_arr)):
                rot_diff = (rot_arr[i][1] - rot_arr[i - 1][1])
                # euler_rot_diff=(rot_arr[i][0]-rot_arr[i-1][0])
                # Turning right
                if steering > 0:
                    assert rot_diff > 0
                else:
                    assert rot_diff <= 0
            logger.info(
                "Steering -1 and 1 rotates the agent in opposite directions")

    # Tests whether the agent moves in the correct forward/backward directions given positive and negative throttle
    def test_throttle(self, request, env_config):
        logger.info(f"=========== Running {request.node.name}")
        position_diff_arr = []
        throttles = [0.1, 0.5, 1]
        for throttle in throttles:
            env = gym.make("arora-v0", env_config=env_config)
            steering = 0

            logger.debug(f'Throttle is: {throttle}')
            position_arr = []
            o = env.reset()
            for x in range(0, 10):
                o, r, done, i = env.step([throttle, steering, -1])
                pos_now = o[0][:3]
                # logger.info(pos_now)
                position_arr.append(pos_now)

            env.close()
            del env

            avg_position_diff = 0
            for i in range(1, len(position_arr)):
                position_diff = np.linalg.norm(
                    position_arr[i] - position_arr[i - 1])
                logger.debug(position_diff)
                avg_position_diff += position_diff
                if throttle > 0:
                    assert position_diff > 0
                else:
                    assert position_diff <= 0
            avg_position_diff /= len(position_arr)
            position_diff_arr.append(avg_position_diff)

        logger.info(position_diff_arr)
        logger.info(
            f'Each throttle {throttles} produce the above average postion diff')


if __name__ == '__main__':
    pytest.main([__file__])
