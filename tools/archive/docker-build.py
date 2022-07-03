#!/usr/bin/env python3

import argparse
import subprocess
from subprocess import run, DEVNULL
import shutil, os

with open('../version.txt', 'r') as vf:
    __version__ = vf.read().strip()

def image_exists(iname):
    cp = run(['docker', 'image', 'inspect', iname], stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL)
    return True if cp.returncode == 0 else False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--remote', action='store_true')
    parser.add_argument('-l', '--local', action='store_true')
    parser.add_argument('-s', '--singularity', action='store_true')
    args = parser.parse_args()

    iname = f"navsim:{__version__}"  # image name
    rrepo = "ghcr.io/armando-fandango"  # image repo
    rname = f"{rrepo}/{iname}"
    lrepo = "localhost:5000"
    lname = f"{lrepo}/{iname}"

    if image_exists(iname):
        print(f"found image {iname}.. rebuilding")
    else:
        print(f"creating image {iname}")

    bopt = "--no-cache"
    if os.path.exists('repo.zip'):
        os.remove('repo.zip')
    cp = subprocess.run(['zip','-r','tools/repo.zip','navsim-lab','navsim-envs','navsim-mlagents/ml-agents-envs',
                         'navsim-mlagents/gym-unity','version.txt',
                         '-x','"*/\build"',
                         '-x','"*/\dist"',
                         '-x','"*/\*pycache*"',
                         '-x','"*/\*.egg-info"'
                         ], cwd='..')
    cp = subprocess.run(['docker', 'build', bopt, '-t', iname, '.'])
    if os.path.exists('repo.zip'):
        os.remove('repo.zip')
    if cp.returncode==0:
        subprocess.run(['docker', 'tag', iname, lname])
        subprocess.run(['docker', 'tag', iname, rname])
        print(f"created image {iname} aka {lname} aka {rname}")
        if args.remote:
            subprocess.run(['docker', 'push', rname])
        if args.local:
            subprocess.run(['docker', 'push', lname])
        if args.singularity:
            print('headless singularity build not implemented yet')
    else:
        print(f"Failed to create image {iname}")


if __name__ == "__main__":
    main()
#
# docker run -d -p 5000:5000 --restart=always --name registry -v /mnt/data/docker-registry:/var/lib/registry registry:2
# docker container stop registry && docker container rm -v registry