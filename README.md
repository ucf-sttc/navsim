# Introduction 
A navigation simulator (navsim) API built on top of Python, Pytorch.

In the future, navsim may be compatible with a variety of simulators, but for now it uses A Realistc Open environment for Rapid Agent training(ARORA) Simulator, that is a highly attributed Unity3D GameEngine based Berlin city environment.

You can either use navsim [with container](#use-navsim-with-container) or [without container](#use-navsim-without-container).

# Get the code

clone the `navsim` repo:

```sh
git clone --recurse-submodules git@github.com:ucf-sttc/navsim.git
```

# Use navsim inside container

## Install Pre-requisites for using inside container

* Install [nvidia driver](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver)
* Install [docker](https://docs.docker.com/get-docker/)
* Install [nvidia container toolkit](https://github.com/NVIDIA/nvidia-docker)

## Initial setup

  Use paths specific to your system and update the lines 4-7 of `navsim/tools/docker-compose.yml`.

  * `~exp/` : Experiments are run in this folder
  * `~/unity-envs/` : Unity-based standalone binaries are kept here
  * `~/workspaces/` : Navsim code folder is kept here
  * `/data`: This is where all the above symlinked folders are present 


    Please note in line 7 of `docker-compose.yml`: because on our systems, all our folders such as `workspaces`, `exp ` and `unity-envs` reside in `/data` and are symlinked in home folder, hence this `/data` folder also has to be mounted else the symlinks wont work in container. If you are not using symlinks, then you can remove this line.

  Run the following command to test everything works fine:

  `docker compose run --rm navsim-1`

  `docker compose run --rm navsim-1 navsim --help`

## Run the experiments (inside container)

  Run the following command: (rremove `-d` after `run` to run it in foreground):
  
  `docker compose run -d navsim-1 <navsim command>`

# Use navsim in the host (without container)

## Install pre-requisites for using in the host (without container)

* Install `mamba 4.12.0-3` : `https://github.com/conda-forge/miniforge/releases/download/4.12.0-3/Mambaforge-4.12.0-3-Linux-x86_64.sh`
* ```sh
  cd /path/to/navsim/repo
  mamba create -n navsim python==3.8.16
  mamba env update -n navsim -f tools/pyenv/navsim.yml
  conda activate navsim
  ./install-repo.sh
  ```
## Run the experiments (in the host, without container)

* Read `navsim_envs` tutorial to use and test the `navsim_envs`
* Run `jupyter notebook`. The notebooks are in `examples` folder.
* Run the `<navsim command>` described in the section below

# The `<navsim command>`

* `navsim --plan --env arora-v0 --show_visual --env_path ~/unity-envs/ARORA_2.10.17_simpro/ARORA.x86_64`
* `navsim_env_test min_env_config.yml`
* `navsim --help` shows the options
* `navsim --run_id $run_id --env_path $envdir/$envbin` - executes and/or trains the model
* `navsim-benchmark $envdir/$envbin` - benchmarks the model
* `navsim-saturate-gpu $envdir/$envbin` - Saturates the GPU
* Replace the navsim command with your own command if you are just importing
  the NavSim env and have your own code in experiment directory.

```{program-output} navsim --help

```

# Contributing to NavSim API

## General dev info:
* Use only google style to document your code:
  https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google

## to modify and build the docs:

1. Create `<version>-docs` branch from `master` or `version` branch
2. Make sure you have navsim conda env activated and local repo installed with pip.
3. Modify the `.md` files in `/path/to/navsim-lab`, `/path/to/ai_coop_py`, `/path/to/navsim-envs`
4. go to `ai_coop_py/docs`
5. run `make html latexpdf`. If still errors, please call me and I will help fix
6. `pdf` and `html` versions are inside the `docs/build/...` folders
7. If you are happy with the PDF/formatting etc then commit and push the doc branch back

## How to build the container (not needed if you want to use our pre-built containers)

Inside `navsim` repo, follow these commands:

```sh
cd tools
./zip-repo
docker compose build navsim-1 \
  --build-arg duid=1003 \
  --build-arg dgid=1003

docker login ghcr.io -u armando-fandango     # replace with your github login and pat
docker compose push navsim-1
```

## How to fix the id of user inside the container

```sh
cd tools
docker compose build navsim-1-fixid \
  --build-arg from="ghcr.io/ucf-sttc/navsim/navsim:1.0.0-navsim" \
  --build-arg duid=1003 \
  --build-arg dgid=1003
```
