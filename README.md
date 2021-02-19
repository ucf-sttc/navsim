# Introduction 
This is Pytorch and Ml-Agents based front end for learning reinforcement learning based models for visual navigation.

# Getting Started

## To run the singularity container
1. Ask Armando for latest version number of container, e.g. 0.0.1
* Note: Do it on a partiton that has at least 10GB space as the next step will create navsim_0.0.1.sif file of ~10GB.

```
ver=0.0.3
singularity pull docker://ghcr.io/armando-fandango/navsim:$ver
singularity shell --nv navsim_0.0.1.$ver
```
# To run the Docker container:

```
docker pull ghcr.io/armando-fandango/navsim:$ver
docker run -it --gpus all --name navsim_${ver}_1 \
  -u $(id -u):$(id -g) -w ${PWD} \
  -e DISPLAY -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /mnt:/mnt \ 
  navsim_$ver bash
```

## To run NeuralSLAM on Habitat
TODO

## To run NeuralSLAM on BerlinWalk
TODO


# Contribute

Only Armando is building and developing in this repo for now. Once we open development for others, we shall add instructions here.
