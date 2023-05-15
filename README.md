# Introduction 
A navigation simulator (navsim) API built on top of Python, Pytorch.

In the future, navsim may be compatible with a variety of simulators, but for now it uses A Realistc Open environment for Rapid Agent training(ARORA) Simulator, that is a highly attributed Unity3D GameEngine based Berlin city environment.

You can either use navsim [with container](#use-navsim-with-container) or [without container](#use-navsim-without-container).

# Get the code

clone the `navsim` repo:

```sh
git clone --recurse-submodules git@github.com:ucf-sttc/navsim.git
```

All further commands should be done inside navsim repo: `cd navsim`

# Use navsim inside container (preferred and recommended way)

## Install Pre-requisites for using inside container

Please make sure to install the following as per latest instrunctions that work for your system. As a guidance the link to instructions that worked are being provided.
* Install [nvidia driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
* Install [docker engine](https://docs.docker.com/get-docker/) | Do not install docker desktop. It has been tested not to work with desktop.
* Install [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide)

To check that dependencies are working properly for your user:

`docker run --rm --runtime=nvidia --gpus all debian:11.6-slim nvidia-smi`

You should see nvidia-smi output.

## Fix the id of user inside the container <a name="fixid"></a>

The user inside the container, `ezdev`, comes with an id of 1000:1000. If you want this user to be able to read and write files as per your uid:gid, then run the following command to fix the id of the user inside the container:

```sh
docker compose build navsim-1-fixid \
  --build-arg duid=1003 \
  --build-arg dgid=1003
```
This example assumes you want to change the id to 1003:1003. 

You can also specify the image to fix.

```sh
docker compose build navsim-1-fixid \
  --build-arg img="ghcr.io/ucf-sttc/navsim/navsim:1.0.0-navsim" \
  --build-arg duid=1003 \
  --build-arg dgid=1003
```

You will see output similar to following:

```console
armando@thunderbird:~/workspaces/navsim$ docker compose build navsim-1-fixid \
>   --build-arg duid=1003 \
>   --build-arg dgid=1003
[+] Building 5.5s (8/8) FINISHED                                                                                                                                             
 => [internal] load .dockerignore                                                0.1s
 => => transferring context: 2B                                                  0.0s
 => [internal] load build definition from Dockerfile-navsim-fixid                0.0s
 => => transferring dockerfile: 440B                                             0.0s
 => [internal] load metadata for ghcr.io/ucf-sttc/navsim/navsim:1.0.0-navsim     0.0s
 => [1/4] FROM ghcr.io/ucf-sttc/navsim/navsim:1.0.0-navsim                       1.1s
 => [2/4] RUN id ezdev                                                           0.5s
 => [3/4] RUN usermod -u 1003 ezdev && groupmod -g 1003 ezdev                    2.5s
 => [4/4] RUN id ezdev                                                           0.7s
 => exporting to image                                                           0.4s
 => => exporting layers                                                          0.3s
 => => writing image sha256:e69a490b875892bdbb5498797dcef3aa4551223b5309f80d     0.0s
 => => naming to ghcr.io/ucf-sttc/navsim/navsim:1.0.0-navsim                     0.0s
```

## Initial setup

  Modify the lines 4-7 of `navsim/docker-compose.yml` for paths specific to your system.

  * `$HOME/exp/` : Experiments are run in this folder
  * `$HOME/unity-envs/` : Unity-based standalone binaries are kept here
  * `$HOME/workspaces/navsim` : Navsim code folder is kept here
  * `/data`: This is where all the above symlinked folders are present. Please note in line 7 of `docker-compose.yml`: because on our systems, all our folders such as `workspaces/navsim`, `exp ` and `unity-envs` reside in `/data` and are symlinked in home folder, hence this `/data` folder also has to be mounted else the symlinks wont work in container. If you are not using symlinks, then you can remove this line.

  Run the following command to test everything works fine:

  `docker compose run --rm navsim-1`

  `docker compose run --rm navsim-1 navsim --help`

## Run the experiments (inside container)

  Run the following command: (remove `-d` after `run` to run it in foreground):
  
  `docker compose run -d --rm navsim-1 <navsim command>`

# Use navsim in the host (without container)

## Install pre-requisites for using in the host (without container)

* Install `mamba 4.12.0-3` : `https://github.com/conda-forge/miniforge/releases/download/4.12.0-3/Mambaforge-4.12.0-3-Linux-x86_64.sh`
* ```sh
  cd /path/to/navsim/repo
  mamba create -n navsim python==3.8.16
  mamba env update -n navsim -f tools/pyenv/navsim.yml
  conda activate navsim
  ./install-repo.sh
  ```
## Run the experiments (in the host, without container)

* Read `navsim_envs` tutorial to use and test the `navsim_envs`
* Run `jupyter notebook`. The notebooks are in `examples` folder.
* Run the `<navsim command>` described in the section below

# The `<navsim command>`

* `navsim --plan --env arora-v0 --show_visual --env_path /unity-envs/<path-to-arora-binary>`. For example, `<path-to-arora-binary>` in our case is the foldername and binary after `$HOME/unity-envs` that we mapped in line 5 of docker-compose earlier: `ARORA/ARORA.x86_64`.
* `navsim_env_test min_env_config.yml`
* `navsim --help` shows the options
* `navsim --run_id $run_id --env_path $envdir/$envbin` - executes and/or trains the model
* `navsim-benchmark $envdir/$envbin` - benchmarks the model
* `navsim-saturate-gpu $envdir/$envbin` - Saturates the GPU
* Replace the navsim command with your own command if you are just importing
  the NavSim env and have your own code in experiment directory.

```{program-output} navsim --help

```

# Errors FAQ

## Getting vulkan error while starting the container
```console
ERROR: [Loader Message] Code 0 : /usr/lib/x86_64-linux-gnu/libvulkan_radeon.so: cannot open shared object file: No such file or directory
No protocol specified
No protocol specified
ERROR: [Loader Message] Code 0 : loader_scanned_icd_add: Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0
ERROR: [Loader Message] Code 0 : /usr/lib/i386-linux-gnu/libvulkan_intel.so: cannot open shared object file: No such file or directory
ERROR: [Loader Message] Code 0 : /usr/lib/i386-linux-gnu/libvulkan_radeon.so: cannot open shared object file: No such file or directory
ERROR: [Loader Message] Code 0 : /usr/lib/x86_64-linux-gnu/libvulkan_intel.so: cannot open shared object file: No such file or directory
ERROR: [Loader Message] Code 0 : /usr/lib/i386-linux-gnu/libvulkan_lvp.so: cannot open shared object file: No such file or directory
ERROR: [Loader Message] Code 0 : /usr/lib/x86_64-linux-gnu/libvulkan_lvp.so: cannot open shared object file: No such file or directory
Cannot create Vulkan instance.
This problem is often caused by a faulty installation of the Vulkan driver or attempting to use a GPU that does not support Vulkan.
ERROR at /build/vulkan-tools-oFB8Ns/vulkan-tools-1.2.162.0+dfsg1/vulkaninfo/vulkaninfo.h:666:vkCreateInstance failed with ERROR_INCOMPATIBLE_DRIVER
```

Solution: For fixing this error you have to update your nvidia driver and fix the id inside the container, as follows:

1. Check your nvidia driver with the following commands: `sudo apt list --installed | grep nvidia-driver` and `nvidia-smi`

For example on our laptop:

```console
armando@thunderbird:~/workspace/navsim$ sudo apt list --installed | grep nvidia-driver

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

nvidia-driver-470/focal-updates,focal-security,now 470.182.03-0ubuntu0.20.04.1 amd64 [installed]
```
```console
armando@thunderbird:~/workspace/navsim$ nvidia-smi
Sun May 14 10:53:30 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.182.03   Driver Version: 470.182.03   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
```
Reinstall the nvidia-driver or update it to latest one.
```sh
sudo apt update
sudo apt install nvidia-driver-530
sudo reboot
```
If you dont rebooot after installing the driver then you will get the following error: 
```console
Failed to initialize NVML: Driver/library version mismatch
```

2. Update the id inside the container as per section: [Fix the id of user inside the container](#fixid)


# Contributing to NavSim API

## General dev info:
* Use only google style to document your code:
  https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google

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

```sh
./zip-repo
docker compose build navsim-1-build
```
