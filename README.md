# Introduction 
This is Pytorch and Ml-Agents based front end for learning reinforcement learning based models for visual navigation.

# Run the container first

## To run the singularity container
Note: Do it on a partition that has at least 10GB space as the next step will create navsim_0.0.1.sif file of ~10GB.

```
ver=0.0.3
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