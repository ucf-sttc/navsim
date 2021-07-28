import pytest
import click

def main():
    pytest.main(['-q','-s','--pyargs', 'navsim_envs'])


# For python debugger to directly run this script
if __name__ == "__main__":
    main()
