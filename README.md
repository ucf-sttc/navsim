# Introduction 
A navigation simulator API built on top of Python, Stable Baselines 3, Pytorch.

Can use many simulator backends, for now uses the Aurora Simulator, that is a 
Unity3D GameEngine based Berlin city environment.

# Pre-requisites

Following should be pre-installed on the host machine:
* nvidia driver https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver
* docker  https://docs.docker.com/get-docker/
* nvidia container toolkit https://github.com/NVIDIA/nvidia-docker

# Versions

There are three components: navsim binary, navsim python api, navsim container
You can use any version of each of them as long as first two digits match. These are latest releases of each of them:
* binary 2.8.1
* python api 2.8.1
* container 2.8.1

# How to use the navsim env 

Assuming your code is in a folder defined in environment variable `expdir`. 
In your code import `NavSimGymEnv` from the `navsim` package. Either use it to
instantiate env objects or extend it by subclassing.

Follow the instructions in "how to run the navsim training" section below, 
replace `navsim` command with your own command, for example: `my-training`

# How to run the navsim training

You can either run directly on a host machine or in a container. 

## Option 1: Container

1. Download and extract the unity binary zip file, and 
set the following, after changing first two lines for your system:
```
envdir=$(realpath "/data/work/unity-envs/Build2.8.1"); envbin="Berlin_Walk_V2.x86_64"; expdir=$(realpath "$HOME/exp");
repo="ghcr.io/armando-fandango"; cname="$(hostname)_navsim_1"
```
2. Run the container:
```
cd $expdir
docker run --rm --privileged -it --runtime=nvidia \
--name $cname \
-h $cname \
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
$repo/navsim:2.8.1 DISPLAY=:0.0 <navsim command>
```

### The Variable `DISPLAY=:0.0`
The display variable points to X Display server, and takes a value of `hostname:D.S`, where:
* `hostname` can be empty.
* `D` refers to the display index, which is 0 generally.
* `S` refers to the screen index, which is 0 generally but in a GPU based system, each GPU might be connected to a different screen. In our container, this number refers to the GPU on which the environment binary will run.

For the purpose of navsim container, use `DISPLAY=0.0` and change the last zero to the index number if GPU where environment binary can run.

### The `<navsim command>`

* `navsim --env $envdir/$envbin` - executes and/or trains the model
* `navsim-benchmark $envdir/$envbin` - benchmarks the model
* `navsim-saturate-gpu $envdir/$envbin` - Saturates the GPU
* Replace the navsim command with your own command if you are just importing the NavSim env.

## Option 2: TODO: Run on host directly
### Fix the following parts of readme Headless Run with X-Server 

Assumption: X is installed, nvidia-drivers

Install tmux (useful for persistence and shell management) (Cheat Sheet: https://gist.github.com/MohamedAlaa/2961058)  

For tmux hotkeys press ctrl+b then following key  

* Start tmux session: tmux new -s <session name>
* Open another tmux shell: ctrl + b, % (vertical pane) Or ctrl + b, " (horizontal pane)
* Move between panes: ctrl + <left, right, up, down>
* Detach from tmux session: ctrl + b, d  (detach from tmux session)
* Attach to existing tmux session: tmux attach -t <session name>
* Exit Session: Type exit into all open shells within session

### TODO: To run the singularity container
Note: Do it on a partition that has at least 10GB space as the next step will create navsim_0.0.1.sif file of ~10GB.

singularity pull docker://$repo/navsim:$ver
singularity shell --nv \
-B <absolute path of sim binary folder> # not needed if path to binary is inside $HOME folder  
-B <absolute path of current folder> # not needed if path to current folder is inside $HOME folder
navsim_$ver.sif


For IST Devs: From local docker repo for development purposes:

SINGULARITY_NOHTTPS=true singularity pull docker://$repo/navsim:$ver
