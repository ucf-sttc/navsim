import psutil
import sys


def main():
    env_kill(sys.argv.pop())


def env_kill(env_filename):
    for proc in psutil.process_iter():
        # check whether the process name matches
        if env_filename in proc.name():
            proc.kill()


# For python debugger to directly run this script
if __name__ == "__main__":
    main()
