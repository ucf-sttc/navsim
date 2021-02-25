# Introduction 
This is Pytorch and Ml-Agents based front end for learning reinforcement learning based models for visual navigation.

# Run the container first

## Headless Run with X-Server 

Assumption: X is installed, nvidia-drivers

Install tmux (useful for persistence and shell management) (Cheat Sheet: https://gist.github.com/MohamedAlaa/2961058)  

For tmux hotkeys press ctrl+b then following key  

* Start tmux session: tmux new -s <session name>
* Open another tmux shell: ctrl + b, % (vertical pane) Or ctrl + b, " (horizontal pane)
* Move between panes: ctrl + <left, right, up, down>
* Detach from tmux session: ctrl + b, d  (detach from tmux session)
* Attach to existing tmux session: tmux attach -t <session name>
* Exit Session: Type exit into all open shells within session


### X server on host setup and startup (Admin required)
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

## X server in container
```
./docker-run.sh --xserver
```

### GPU Usage examples
Note: In new pane run container (can pass DISPLAY below as a variable), navigate to training command directory, and setup training specific configurations
```
Run a command on a specific gpu

DISPLAY=:88.<screen idx> navsim 
DISPLAY=:88.0 navsim-benchmark AICOOP_binaries/Build2.4.4/Berlin_Walk_V2.x86_64 -a VectorVisual
DISPLAY=:88.3 navsim-saturate-gpu AICOOP_binaries/Build2.4.4/Berlin_Walk_V2.x86_64  
```


## To run the singularity container
Note: Do it on a partition that has at least 10GB space as the next step will create navsim_0.0.1.sif file of ~10GB.

```
ver=1.0.0
singularity pull docker://ghcr.io/armando-fandango/navsim:$ver
singularity shell --nv navsim_$ver.sif
```
From local docker repo:
```
ver=0.0.3; SINGULARITY_NOHTTPS=true singularity pull docker://localhost:5000/navsim:$ver
```
## To run the Docker container:

```
docker pull ghcr.io/armando-fandango/navsim:$ver
docker run -it --gpus all --name navsim_${ver}_1 \
  -u $(id -u):$(id -g) -w ${PWD} \
  -e DISPLAY -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /mnt:/mnt \ 
  navsim_$ver bash
```
# Now run BerlinWalk
* `navsim` - executes and/or trains the model
* `navsim-benchmark` - benchmarks the model
* `navsim-saturate-gpu` - Saturates the GPU

# Contribute

Send a pull request
