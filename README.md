# Introduction 
A navigation simulator (navsim) API built on top of Python, Pytorch.

In the future, navsim may be compatible with a variety of simulators, but for now it uses A Realistc Open environment for Rapid Agent training(ARORA) Simulator, that is a highly attributed Unity3D GameEngine based Berlin city environment.

You can either use navsim [with container](#use-navsim-with-container) or [without container](#use-navsim-without-container).

# Getting the code

clone the `navsim` repo:

```
git clone --recurse-submodules git@github.com:ucf-sttc/navsim.git
```
or
```
git clone git@github.com:ucf-sttc/navsim.git
cd navsim
git submodule update --init --recursive
```


# use navsim with container

## Pre-requisites

* [nvidia driver](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver)
* [docker](https://docs.docker.com/get-docker/)
* [nvidia container toolkit](https://github.com/NVIDIA/nvidia-docker)

## How to run the binary in container

* In your home folder have the following folders ready:
  * `~/exp/` : Experiments are run in this folder
  * `~/unity-envs/` : Unity-based standalone binaries are kept here

  If you use other folder names then change in the following commands appropriately

* copy `navsim/tools/docker-compose.yml` to `~/exp`

* Because on our systems, the `exp ` and `unity-envs` reside in `/data` and are symlinked in home folder, hence this `/data` folder also has to be mounted else the symlinks wont work in container. 

  To mount the folder in your system, in line 5 of `~/exp/docker-compose.yml`, change `/data:/data` to `/whatever-your-folder:/whatever-your-folder`

* For sim-pro binary (remove -d after run if you dont want to run it in background):

  ```
  DUID="$(id -u)" DGID="$(id -g)" docker-compose run -d navsim-headless-ubuntu2004 <navsim command>
  ```

* For non-simpro binary (remove -d after run if you dont want to run it in background):

  ```
  DUID="$(id -u)" DGID="$(id -g)" docker-compose run -d navsim-headfull-ubuntu2004 <navsim command>
  ```

# use navsim without container

## Pre-requisites

* Either setup `navsim` conda env or activate your own python virtual environment:
  * Simple way to create conda env: `conda create -n navsim python=3.8 jupyter && conda activate navsim`
  * or install conda env from our repo: Go to `navsim` conda section
* Install the repos in `navsim` virtual env
  ```
  cd /path/to/navsim
  ./install-repo.sh
  ```

## How to run the binary without container

* Read `navsim_envs` tutorial to use and test the `navsim_envs`
* Run `jupyter notebook`. The notebooks are in `examples` folder.
* Run the `<navsim-command>` described in the section below

# The `<navsim command>`

* `navsim --plan --env arora-v0 --env_path ~/unity-envs/ARORA_2.10.17_simpro/ARORA.x86_64`
* `navsim_env_test min_env_config.yml`
* `navsim --help` shows the options
* `navsim --run_id $run_id --env_path $envdir/$envbin` - executes and/or trains the model
* `navsim-benchmark $envdir/$envbin` - benchmarks the model
* `navsim-saturate-gpu $envdir/$envbin` - Saturates the GPU
* Replace the navsim command with your own command if you are just importing
  the NavSim env and have your own code in experiment directory.

# Contributing to NavSim API

## General dev info:
* Use only google style to document your code:
  https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google

## Setup `navsim` conda env:
1. Go to tools folder where we have the following files:
    ```
    ezai-conda.sh
    ezai-conda-req.txt
    ezai-pip-req.txt
    ```
3. miniconda: We suggest you install miniconda from our script, but if you have
   miniconda installed already then you can skip to next step to create conda
   environment. If next step doesn't work, then come back and follow the
   instructions to install miniconda.
    ```
    CONDA_ROOT=/opt/conda
    sudo mkdir $CONDA_ROOT
    sudo chown $(id -u) $CONDA_ROOT
    source ezai-conda.sh && install_miniconda
    ```
4. Create the conda env for `navsim`
    ```
    source ezai-conda.sh && ezai_conda_create --venv "$(conda info --base)/envs/navsim"
    ```
5. Optional: Install jupyter in `navsim`
    ```
    conda activate navsim && source ezai-conda.sh && install_jupyter
    ```
   
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

```shell
cd tools
./zip-repo
docker-compose build navsim-headless-ubuntu2004
docker-compose build navsim-headfull-ubuntu2004

docker login ghcr.io -u armando-fandango     # replace with your github login and pat
docker push ghcr.io/ucf-sttc/navsim/navsim:0.1-navsim-headfull-ubuntu2004
docker push ghcr.io/ucf-sttc/navsim/navsim:0.1-navsim-headless-ubuntu2004
```
