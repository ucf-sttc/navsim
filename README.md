# Introduction 
A navigation simulator (navsim) API built on top of Python, Pytorch.

In the future, navsim may be compatible with a variety of simulators, but for now it uses A Realistc Open environment for Rapid Agent training(ARORA) Simulator, that is a highly attributed Unity3D GameEngine based Berlin city environment.

# Installation

* clone the `ai_coop_py` repo and checkout the active branch
  ```
  git clone git@ssh.dev.azure.com:v3/emb-ist-edu/AI%20COOP/ai_coop_py
  cd ai_coop_py
  git submodule update --init
  ```
* Either setup `navsim` conda env or activate your own python virtual environment:
  * Simple way to create conda env: `conda create -n navsim python=3.8 jupyter && conda activate navsim`
  * or install conda env from our repo: Go to `navsim` conda section
* Install the repos in `navsim` virtual env
  ```
  cd /path/to/ai_coop_py
  ./install-repo.sh
  ```
* Read `navsim_envs` tutorial to use and test the `navsim_envs`
* Run `jupyter notebook`. The notebooks are in `examples` folder.

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

## navsim-lab on container

### Pre-requisites

* [nvidia driver](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver)
* [docker](https://docs.docker.com/get-docker/)
* [nvidia container toolkit](https://github.com/NVIDIA/nvidia-docker)

### How to run the navsim training

1. Download and extract the unity binary zip file
2. The following environment variables need to be set in both cases:
   ```shell
   envdir=$(realpath "/data/work/unity-envs/Build2.10.13");
   envbin="Berlin_Walk_V2"; 
   expdir=$(realpath "$HOME/exp"); 
   run_id="demo"; 
   repo="ghcr.io/armando-fandango";
   cd $expdir
   ```
3. Now follow the container instructions below:

4. Note: Make sure you are in experiment directory, as container will dump the files there.
   ```shell
   cd $expdir
   docker run --rm --privileged -it --runtime=nvidia \
   --name $run_id \
   -h $run_id \
   -e XAUTHORITY \
   -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
   -e USER_ID=$(id -u) -e USER_HOME="$HOME" \
   -v $HOME:$HOME \
   -v /etc/group:/etc/group:ro \
   -v /etc/passwd:/etc/passwd:ro \
   -v /etc/shadow:/etc/shadow:ro \
   -v $envdir:$envdir \
   -v $expdir:$expdir \
   -w $expdir \
   $repo/navsim:2.10.13 DISPLAY=:0.0 <navsim command>
   ```
Note: Add the following if you want to use your local repos instead of the ones burnt in the container:
`-v $(realpath "$HOME/projects/ai_coop_py") /opt/conda/navsim-repo`

#### The Variable `DISPLAY=:0.0`
The display variable points to X Display server, and takes a value of `hostname:D.S`, where:
* `hostname` can be empty.
* `D` refers to the display index, which is 0 generally.
* `S` refers to the screen index, which is 0 generally but in a GPU based
  system, each GPU might be connected to a different screen.
  In our container, this number refers to the GPU on which the environment
  binary will run.

For the purpose of navsim container, use `DISPLAY=:0.0` and change the last
zero to the index number of GPU for environment binary.


#### The `<navsim command>`

* `navsim_env_test min_env_config.yml`
* `navsim --help` shows the options
* `navsim --run_id $run_id --env_path $envdir/$envbin` - executes and/or trains the model
* `navsim-benchmark $envdir/$envbin` - benchmarks the model
* `navsim-saturate-gpu $envdir/$envbin` - Saturates the GPU

* Replace the navsim command with your own command if you are just importing
  the NavSim env and have your own code in experiment directory.

### How to build the headless container

Inside ai_coop_py repo, follow these commands:

```shell
./zip-repo
cd tools
docker-compose build navsim
docker tag navsim:0.1-navsim-ubuntu2004 ghcr.io/ucf-sttc/navsim/navsim:0.1-navsim-ubuntu2004
docker login ghcr.io -u armando-fandango     # replace with your github login and password
docker push ghcr.io/ucf-sttc/navsim/navsim:0.1-navsim-ubuntu2004
```
DUID="$(id -u)" DGID="$(id -g)" docker-compose run navsim navsim --plan --env arora-v0 --env_path ~/unity-envs/ARORA_2.10.17_simpro/ARORA.x86_64