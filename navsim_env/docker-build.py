import argparse
import subprocess
from subprocess import run, DEVNULL
from .docker_utils import image_exists

__version__ = "0.0.4"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--remote', action='store_true')
    parser.add_argument('-s', '--singularity', action='store_true')
    args = parser.parse_args()
    remote = args.remote
    singularity = args.singularity

    irepo = "ghcr.io/armando-fandango"  # image repo
    iname = f"{irepo}/navsim:{__version__}"  # image name
    lrepo = "localhost:5000"
    lname = f"{lrepo}/navsim:{__version__}"

    if image_exists(iname):
        print(f"found image {iname}.. rebuilding")
    else:
        print(f"creating image {iname}")

    bopt = "--no-cache"
    cp = subprocess.run(['docker', 'build', bopt,'-t',iname,'.'])

    if image_exists(iname):
        print(f"created image {iname}")
        if remote:
            print('remote push not implemented yet')
        else:
            subprocess.run(['docker', 'tag', iname, lname])
            subprocess.run(['docker', 'push', lname])
    else:
        print(f"not created image {iname}")


# "$(docker image inspect $iname > /dev/null 2>&1 && echo 1 || echo '')"

if __name__ == "__main__":
    main()
#
# docker run -d -p 5000:5000 --restart=always --name registry -v /mnt/data/docker-registry:/var/lib/registry registry:2
# docker container stop registry && docker container rm -v registry

# ver=0.0.3; SINGULARITY_NOHTTPS=true singularity pull docker://localhost:5000/navsim:$ver
# singularity pull docker://ghcr.io/armando-fandango/navsim:$ver
# SINGULARITY_NOHTTPS=true singularity pull docker://localhost:5000/navsim:$ver
# singularity shell --bind /mnt --nv navsim_${ver}.sif
# docker run -it --gpus all --name navsim_${ver}_1 \
#      -u $(id -u):$(id -g) -w ${PWD} \
#      -v /mnt:/mnt
#      -e DISPLAY -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all \
#      navsim_${ver} bash
