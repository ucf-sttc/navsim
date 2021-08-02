import pytest
import click


def main():
    #pytest.main(['-s','-q','-ra','--disable-warnings', '--pyargs', 'navsim_envs'])
    #pytest.main(['-s','-q','-ra','--disable-warnings', '--pyargs', 'navsim_envs', '-k', 'test_sample_navigable_point'])
    pytest.main(['-s','-q','-ra','--disable-warnings', '--pyargs', 'navsim_envs', '-k', 'test_set_agent_position'])


# For python debugger to directly run this script
if __name__ == "__main__":
    main()
