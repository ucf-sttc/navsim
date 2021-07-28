import unittest

import gym
import sys
import numpy as np

import navsim_envs
from .configs import env_config, config_banner, load_config

logger = navsim_envs.env.NavSimGymEnv.logger


# This runs first before any class in this module runs
def setUpModule():
    if len(sys.argv) > 1:
        env_config_file = sys.argv.pop()
        env_config_from_file = load_config(env_config_file)
        if env_config_from_file is not None:
            logger.info(f'Updating env_config from {env_config_file}')
            env_config.update(env_config_from_file)

    logger.info(f'\n'
                f'{navsim_envs.__version_banner__}'
                f'{config_banner(env_config, "env_config")}'
                )


# This runs after all class in this module runs
def tearDownModule():
    pass


"""
NavSimGymEnvTests1 : env once per class
NavSimGymEnvTests2 : env once per test
NavSimGymEnvTests3 : env more than once within a test
"""


class NavSimGymEnvTests1(unittest.TestCase):
    """
    env is initialized once per class
    """

    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("navsim-v0", env_config=env_config)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        del cls.env

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # Tests whether the navigable map returned has navigable points
    def test_navigable_map(self):
        logger.info("========Test Navigable Map")

        navigable_map = self.env.get_navigable_map()
        logger.debug(type(navigable_map))
        logger.debug(navigable_map)
        logger.info(f'Navigable cells: {navigable_map.sum()}')
        assert navigable_map.sum() > 1

    # Tests whether the return sample points are navigable accouting to the returned navigable map
    def test_sample_navigable_point(self):
        logger.info("========Test Sample Navigable Point")
        resolution_x = 256
        resolution_y = 256
        navigable_map = self.env.get_navigable_map(resolution_x=resolution_x,
                                                   resolution_y=resolution_y)
        samples = 100000
        for i in range(0, samples):
            sampled_point = self.env.sample_navigable_point(
                resolution_x=resolution_x, resolution_y=resolution_y)
            map_point = navigable_map[sampled_point[0]][sampled_point[1]]
            logger.debug(f'{sampled_point}, {map_point}')
            assert map_point == 1

        logger.info(f'{samples} sampled points are navigable')

    # Tests whether the navmap to unity and back coordinate conversion are correct
    def test_coordinate_conversion(self):
        logger.info("========Test Coordinate Conversion")
        navmap_max_x = 256
        navmap_max_y = 256

        resolution_margin = 15

        unity_loc_original = (0, 0)
        navmap_loc = self.env.unity_loc_to_navmap_loc(*unity_loc_original,
                                                      navmap_max_x=navmap_max_x,
                                                      navmap_max_y=navmap_max_y)
        unity_loc = self.env.navmap_loc_to_unity_loc(*navmap_loc,
                                                     navmap_max_x=navmap_max_x,
                                                     navmap_max_y=navmap_max_y,
                                                     navmap_cell_center=False)
        logger.info("unity to nav to unity")
        logger.info(f'{unity_loc_original}, {navmap_loc}, {unity_loc}')
        assert navmap_loc == (0, 0)
        assert unity_loc == (0, 0)

        unity_loc_original = (0, 2665)
        navmap_loc = self.env.unity_loc_to_navmap_loc(*unity_loc_original,
                                                      navmap_max_x=navmap_max_x,
                                                      navmap_max_y=navmap_max_y)
        unity_loc = self.env.navmap_loc_to_unity_loc(*navmap_loc,
                                                     navmap_max_x=navmap_max_x,
                                                     navmap_max_y=navmap_max_y,
                                                     navmap_cell_center=False)
        logger.info(f'{unity_loc_original}, {navmap_loc}, {unity_loc}')
        assert navmap_loc == (0, 255)
        assert unity_loc[0] == 0 and (
                unity_loc[1] > 2665 - resolution_margin and unity_loc[
            1] < 2665 + resolution_margin)

        unity_loc_original = (3283, 0)
        navmap_loc = self.env.unity_loc_to_navmap_loc(*unity_loc_original,
                                                      navmap_max_x=navmap_max_x,
                                                      navmap_max_y=navmap_max_y)
        unity_loc = self.env.navmap_loc_to_unity_loc(*navmap_loc,
                                                     navmap_max_x=navmap_max_x,
                                                     navmap_max_y=navmap_max_y,
                                                     navmap_cell_center=False)
        logger.info(f'{unity_loc_original}, {navmap_loc}, {unity_loc}')
        assert navmap_loc == (255, 0)
        assert (unity_loc[0] > 3283 - resolution_margin and unity_loc[
            0] < 3283 + resolution_margin) and unity_loc[1] == 0

        unity_loc_original = (3283, 2665)
        navmap_loc = self.env.unity_loc_to_navmap_loc(*unity_loc_original,
                                                      navmap_max_x=navmap_max_x,
                                                      navmap_max_y=navmap_max_y)
        unity_loc = self.env.navmap_loc_to_unity_loc(*navmap_loc,
                                                     navmap_max_x=navmap_max_x,
                                                     navmap_max_y=navmap_max_y,
                                                     navmap_cell_center=False)
        logger.info(f'{unity_loc_original}, {navmap_loc}, {unity_loc}')
        assert navmap_loc == (255, 255)
        assert (unity_loc[0] > 3283 - resolution_margin and unity_loc[
            0] < 3283 + resolution_margin) and (
                       unity_loc[1] > 2665 - resolution_margin and
                       unity_loc[1] < 2665 + resolution_margin)


class NavSimGymEnvTests2(unittest.TestCase):
    """
    env is initialized once per test
    """

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.env = gym.make("navsim-v0", env_config=env_config)

    def tearDown(self):
        self.env.close()
        del self.env


class NavSimGymEnvTests3(unittest.TestCase):
    """
    env is initialized multiple times in each test
    """

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass
        # Tests whether mnutiple runs produces the same observation values

    def test_reproducibility(self):
        logger.info("========Test Reproducibility")
        throttle = 1
        obs_arr = []
        runs = 3
        for outer in range(0, runs):
            env = gym.make("navsim-v0", env_config=env_config)
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
        logger.info(f'{runs} env runs produced equal observations for the first episode')

    # Tests whether the agent rotates left/right give positive and negative steering
    def test_rotation(self):
        logger.info("========Test Rotations")
        for steering in [-1, 1]:
            throttle = 1
            env = gym.make("navsim-v0", env_config=env_config)

            logger.info(f'Steering is: {steering}')
            rot_arr = []
            o = env.reset()
            logger.debug('Rot Y in euler, Rot Y in quaternion')
            for x in range(0, 10):
                o, r, done, i = env.step([throttle, steering, -1])
                rot = o[0][7]
                logger.debug(f'{env.agent_rotation_in_euler[2]}, {rot}')
                rot_arr.append((env.agent_rotation_in_euler[1], rot))

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
            logger.info("Steering -1 and 1 rotates the agent in opposite directions") 

    # Tests whether the agent moves in the correct forward/backward directions given positive and negative throttle
    def test_throttle(self):
        logger.info("========Test Throttle")
        position_diff_arr = []
        throttles = [0.1, 0.5, 1]
        for throttle in throttles:
            env = gym.make("navsim-v0", env_config=env_config)
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
        logger.info(f'Each throttle {throttles} produce the above average postion diff')
        


if __name__ == '__main__':
    unittest.main()
