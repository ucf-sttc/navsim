# Introduction 
A navigation simulator API built on top of Python, Stable Baselines.

Can use many simulator backends, for now uses the Unity bases Berlin bakend.

# How to use the navsim API

You can either run directly on a host machine or in a container. 

## Option 1: Container

1. Download and extract the unity binary zip file, and 
set the following, after changing first two lines for your system:
```
envdir="/data/work/unity-envs/Build2.6.0"
expdir="~/exp"
envdir=$(realpath "$envdir")
expdir=$(realpath "$expdir")
envbin="Berlin_Walk_V2.x86_64"
ver="1.0.5"
repo="ghcr.io/armando-fandango"
```
2. Run the container:
```
cd $expdir
docker run --rm --privileged -it --runtime=nvidia \
--name navsim_1 \
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
$repo/navsim:$ver DISPLAY=:0.0 <navsim command>
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
