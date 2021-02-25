import subprocess
from subprocess import run, DEVNULL

def image_exists(iname):
    cp = run(['docker', 'image', 'inspect', iname],stdin=DEVNULL,stdout=DEVNULL,stderr=DEVNULL)
    return True if cp.returncode == 0 else False

