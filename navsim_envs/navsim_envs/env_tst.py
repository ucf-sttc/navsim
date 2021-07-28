import pytest
import click


def main():
    pytest.main(['-s','-q','-ra','--disable-warnings', '--pyargs', 'navsim_envs'])


# For python debugger to directly run this script
if __name__ == "__main__":
    main()
