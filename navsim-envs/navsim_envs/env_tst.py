import pytest
import click


def main():
    pytest.main(['-s','-q','-ra','--disable-warnings', '--pyargs', 'navsim_envs'])
    #pytest.main(['-s','-q','-ra','--disable-warnings', '--pyargs', 'navsim_envs', '-k', 'test_rotation'])
    #pytest.main(['-s','-q','-ra','--disable-warnings', '--pyargs', 'navsim_envs', '-k', 'test_coordinate_conversion'])
    #pytest.main(['-s','-q','-ra','--disable-warnings', '--pyargs', 'navsim_envs', '-k', 'test_set_agent_position'])
    #pytest.main(['-s','-q','-ra','--disable-warnings', '--pyargs', 'navsim_envs', '-k', 'test_set_agent_rotation'])
    #pytest.main(['-s','-q','-ra','--disable-warnings', '--pyargs', 'navsim_envs', '-k', 'test_get_obs'])
    #pytest.main(['-s','-q','-ra','--disable-warnings', '--pyargs', 'navsim_envs', '-k', 'test_reproducibility'])


# For python debugger to directly run this script
if __name__ == "__main__":
    main()
