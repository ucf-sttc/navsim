# Introduction 
A navigation simulator API built on top of Python, Stable Baselines 3, Pytorch.

Can use many simulator backends, for now uses the Aurora Simulator, that is a 
Unity3D GameEngine based Berlin city environment.

# How to use the navsim env

If you only want to use `NavSimGymEnv`, then all you need to do is install 
`navsim` from pip and then either subclass it or use it as follows, or check 
the detailed tutorial <insert link here):

```python
import navsim
import gym

env_config = navsim.util.ObjDict({
        "env_path": "/data/work/unity-envs/Build2.9.2/Berlin_Walk_V2.x86_64",
        "log_folder":"./env_log", 
        "task": 0,
        "goal": 0,
        "goal_distance": 50,
        "reward_for_goal": 50,
        "reward_for_no_viable_path":-50,
        "reward_step_mul": 0.1,
        "reward_collision_mul": 4,
        "reward_spl_delta_mul": 1,
        "agent_car_physics": 0,   
        "debug":False,
        "obs_mode":0,
        "seed":123,
        "save_vector_obs":True,
        "save_visual_obs":True
    })
    
env = gym.make("navsim-v0", env_config=env_config) 
# or use the following method to create an env
env = navsim.NavSimGymEnv(env_config)
```

If you want to use our `navsim` conda environment or `navsim` container then 
follow the instructions below.

# How to use the navsim conda env or container

## Pre-requisites

Following should be pre-installed on the host machine:

### For running inside the containers

* [nvidia driver](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver)
* [docker](https://docs.docker.com/get-docker/)
* [nvidia container toolkit](https://github.com/NVIDIA/nvidia-docker)

### For directly running on the host
  
* X-window system  
* nvidia drivers  
  
## Versions

There are three components: navsim binary, navsim python api, navsim container
You can use any version of each of them as long as first two digits match. 
These are latest releases of each of them:  
* binary 2.10.x  
* python api 2.10.x  
* container 2.10.x  

## How to run the navsim training

You can either run directly on a host machine or in a container. 
If you are running on a host directly, 
first follow the instructions to setup the host.

1. Download and extract the unity binary zip file
2. The following environment variables need to be set in both cases:
   ```shell
   envdir=$(realpath "/data/work/unity-envs/Build2.10.1-dev");
   envbin="Berlin_Walk_V2.x86_64"; 
   expdir=$(realpath "$HOME/exp"); 
   run_id="demo"; 
   repo="ghcr.io/armando-fandango";
   cd $expdir
   ```
3. Now follow the container, or the host option below.
   
### Option 1: Container

Note: Make sure you are in experiment directory, 
as container will dump the files there.

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
   $repo/navsim:2.10.0 DISPLAY=:0.0 <navsim command>
   ```

#### The Variable `DISPLAY=:0.0`
The display variable points to X Display server, and takes a value of `hostname:D.S`, where:
* `hostname` can be empty.
* `D` refers to the display index, which is 0 generally.
* `S` refers to the screen index, which is 0 generally but in a GPU based system, each GPU might be connected to a different screen. In our container, this number refers to the GPU on which the environment binary will run.

For the purpose of navsim container, use `DISPLAY=:0.0` and change the last zero to the index number if GPU where environment binary can run.

### Option 2: Run on host directly - doesn't run headless.

To run on the host, activate the `navsim` virtual environment, only once, 
with following command: `conda activate navsim || source activate navsim`.

Now the navsim env should be activated. If not then go to host setup steps 
and troubleshoot.

Run the `navsim` command as described in its section below.

### The `<navsim command>`

* `navsim --help` shows the options
* `navsim --run_id $run_id --env $envdir/$envbin` - executes and/or trains the model
* `navsim-benchmark $envdir/$envbin` - benchmarks the model
* `navsim-saturate-gpu $envdir/$envbin` - Saturates the GPU
* Replace the navsim command with your own command if you are just importing 
  the NavSim env and have your own code in experiment directory.
  
## Setup the host to run directly
### Assumptions
* Following are installed: X, nvidia drivers

### Steps

1. Download following files:
    * `ezai-conda.sh`
    * `ezai-conda-req.txt`
    * `ezai-pip-req.txt`
2. miniconda: We suggest you install miniconda from our script, but if you have 
   miniconda installed already then you can skip to next step to create conda 
   environment. If next step doesn't work, then come back and follow the 
   instructions to install miniconda.
    ```
    CONDA_ROOT=/opt/conda
    sudo mkdir $CONDA_ROOT
    sudo chown $(id -u) $CONDA_ROOT
    source ezai-conda.sh && install_miniconda
    ```
3. Create the conda env for `navsim`
    ```
    ENVS_ROOT=$(conda info --base)/envs
    source ezai-conda.sh && ezai_conda_create --venv "$ENVS_ROOT/navsim"
    ```

## TODO: Clean up the following section

For tmux hotkeys press ctrl+b then following key  

* Start tmux session: tmux new -s <session name>
* Open another tmux shell: ctrl + b, % (vertical pane) Or ctrl + b, " (horizontal pane)
* Move between panes: ctrl + <left, right, up, down>
* Detach from tmux session: ctrl + b, d  (detach from tmux session)
* Attach to existing tmux session: tmux attach -t <session name>
* Exit Session: Type exit into all open shells within session

## TODO: To run the singularity container
Note: Do it on a partition that has at least 10GB space as the next step will create navsim_0.0.1.sif file of ~10GB.

singularity pull docker://$repo/navsim:$ver
singularity shell --nv \
-B <absolute path of sim binary folder>  not needed if path to binary is inside $HOME folder  
-B <absolute path of current folder>  not needed if path to current folder is inside $HOME folder
navsim_$ver.sif


For IST Devs: From local docker repo for development purposes:

SINGULARITY_NOHTTPS=true singularity pull docker://$repo/navsim:$ver
