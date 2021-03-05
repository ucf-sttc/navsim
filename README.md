# Introduction 
A navigation simulator API built on top of Python, Stable Baselines.

Can use many simulator backends.

# How to use the navsim API

You can either run directly on a host machine or in a container. 

Step 1 is to run the container, or the required services in the host machine.
Step 2 is to run the navsim itself.

## Step 1: Run the container, or the services in host machine

### Step 1 Option 1: Container
First, either start a singularity or a docker container.

Start by defining which repo and version to use.
```
ver=1.0.1
repo="ghcr.io/armando-fandango"
```
For IST Devs: From local docker repo for development purposes:
```
repo="localhost:5000"
```
Next run the container:
#### To run the singularity container
Note: Do it on a partition that has at least 10GB space as the next step will create navsim_0.0.1.sif file of ~10GB.

```
singularity pull docker://$repo/navsim:$ver
singularity shell --nv navsim_$ver.sif
```
For IST Devs: From local docker repo for development purposes:
```
SINGULARITY_NOHTTPS=true singularity pull docker://$repo/navsim:$ver
```
#### To run the Docker container:

```
docker pull $repo/navsim:$ver
docker run --privileged -it --gpus all --name navsim_${ver}_1 \
  -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /mnt:/mnt \
  -v /etc/group:/etc/group:ro \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/shadow:/etc/shadow:ro \
  $repo/navsim:$ver bash
```

### Step 1 Option 2 How to run on host machine without container - TODO
### Please skip to step 2 as these instructions are in progress.
#### X server on host setup and startup (Admin required)

Note: Either run X in a tmux session or have admin start X with generated config and display port in background manually or on startup
 tmux new -s x-server
```
#For Lambda quad
 nvidia-xconfig -o headlessxorg.conf -a --use-display-device=None --virtual=1280x1024
#For V100 servers or Nvidia GRID enabled machines
 nvidia-xconfig -o headlessxorg.conf -a 

#You can select a perferred DISPLAY port
 sudo X -config headlessxorg.conf :88 (Running in background with & stops the process, TODO containerized)
```

## Step 2: Run the navsim
* `navsim --env <path to env file>` - executes and/or trains the model
* `navsim-benchmark <path to env file>` - benchmarks the model
* `navsim-saturate-gpu <path to env file>` - Saturates the GPU

### TODO: This section needs to be fixed - GPU Usage examples
Note: In new pane run container (can pass DISPLAY below as a variable), navigate to training command directory, and setup training specific configurations
```
Run a command on a specific gpu

DISPLAY=:88.<screen idx> navsim 
DISPLAY=:88.0 navsim-benchmark AICOOP_binaries/Build2.4.4/Berlin_Walk_V2.x86_64 -a VectorVisual
DISPLAY=:88.3 navsim-saturate-gpu AICOOP_binaries/Build2.4.4/Berlin_Walk_V2.x86_64  
```

## TODO: Fix the following parts of readme Headless Run with X-Server 

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
