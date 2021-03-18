# Introduction 
A navigation simulator API built on top of Python, Stable Baselines.

Can use many simulator backends, for now uses the Unity bases Berlin bakend.

# How to use the navsim API

You can either run directly on a host machine or in a container. 

## Option 1: Container

1. Download and extract the unity environment binary zip file, and 
set the following:
```
envdir="$HOME/unity-envs/BerlinLinux_2020.2.1f1_R12_2.4.5"
envbin="Berlin_Walk_V2.x86_64"
```
2. Define which repo and version to use.
```
ver="1.0.4"
repo="ghcr.io/armando-fandango"
```
For IST Devs: From local docker repo for development purposes:
```
repo="localhost:5000"
```
3. Next run the container:

### TODO: To run the singularity container
Note: Do it on a partition that has at least 10GB space as the next step will create navsim_0.0.1.sif file of ~10GB.

```
singularity pull docker://$repo/navsim:$ver
singularity shell --nv \
  -B <absolute path of sim binary folder> # not needed if path to binary is inside $HOME folder  
  -B <absolute path of current folder> # not needed if path to current folder is inside $HOME folder
  navsim_$ver.sif
```
For IST Devs: From local docker repo for development purposes:
```
SINGULARITY_NOHTTPS=true singularity pull docker://$repo/navsim:$ver
```
### To run the Docker container:

```
docker pull $repo/navsim:$ver
docker run --rm --privileged -it --runtime=nvidia \
  --name navsim_$ver_1 \
  -e XAUTHORITY \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e USER_ID=$(id -u) -e USER_HOME="$HOME" \
  -v $HOME:$HOME \
  -v /etc/group:/etc/group:ro \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/shadow:/etc/shadow:ro \
  -v "$(realpath $envdir)":$HOME/rlenv \
  -v "$(realpath $PWD)":$HOME/$(basename $PWD) \
  -w $HOME/$(basename $PWD) \
  $repo/navsim:$ver <navsim_command>
```

### The `<navsim_command>`
* `DISPLAY=:0.0 navsim --env $HOME/rlenv/$envbin` - executes and/or trains the model
* `DISPLAY=:0.0 navsim-benchmark $HOME/rlenv/$envbin` - benchmarks the model
* `DISPLAY=:0.0 navsim-saturate-gpu $HOME/rlenv/$envbin` - Saturates the GPU

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


# Contribute

Send a pull request to Armando Fandango
