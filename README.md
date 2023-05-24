A navigation simulator (navsim) API built on top of Python, Pytorch.

In the future, navsim may be compatible with a variety of simulators, but for now it uses A Realistc Open environment for Rapid Agent training(ARORA) Simulator, that is a highly attributed Unity3D GameEngine based Berlin city environment.

# Getting started

You can either run navsim [inside container](#run-navsim-inside-container-preferred-and-recommended-way) or [directly in host without container](#run-navsim-in-the-host-without-container).

## Get the code

clone the `navsim` repo:

```sh
git clone --recurse-submodules git@github.com:ucf-sttc/navsim.git
```

All further commands should be done inside navsim repo: `cd navsim`

## Run navsim inside container (preferred and recommended way)

### Install Pre-requisites for using inside container

Please make sure to install the following as per latest instrunctions that work for your system. As a guidance the link to instructions that worked are being provided.
* Install [nvidia driver](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
* Install [docker engine](https://docs.docker.com/get-docker/) | Do not install docker desktop. It has been tested not to work with desktop.
* Install [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide)

To check that dependencies are working properly for your user:

`docker run --rm --runtime=nvidia --gpus all debian:11.6-slim nvidia-smi`

You should see nvidia-smi output.

### Fix the user id inside the container
The user inside the container, `ezdev`, comes with an id of 1000:1000. If you want this user to be able to read and write files as per your uid:gid, then run the following command to fix the id of the user inside the container:

```sh
DUID=`id -u` DGID=`id -g` docker compose build navsim-fixid
```

You can also specify the image to fix as env variable:

```sh
IMAGE=ghcr.io/ucf-sttc/navsim/navsim:<version> DUID=`id -u` DGID=`id -g` docker compose build navsim-fixid
```

You will see output similar to following:

```console
armando@thunderbird:~/workspaces/navsim$ docker compose build navsim-fixid
[+] Building 5.5s (8/8) FINISHED                                                                                                                                             
 => [internal] load .dockerignore                                      0.1s
 => => transferring context: 2B                                        0.0s
 => [internal] load build definition from Dockerfile-navsim-fixid      0.0s
 => => transferring dockerfile: 440B                                   0.0s
 => [internal] load metadata for ghcr.io/ucf-sttc/navsim/navsim        0.0s
 => [1/4] FROM ghcr.io/ucf-sttc/navsim/navsim                          1.1s
 => [2/4] RUN id ezdev                                                 0.5s
 => [3/4] RUN usermod -u 1003 ezdev && groupmod -g 1003 ezdev          2.5s
 => [4/4] RUN id ezdev                                                 0.7s
 => exporting to image                                                 0.4s
 => => exporting layers                                                0.3s
 => => writing image sha256:e69a490b875892bdbb5498797dcef3aa4551223    0.0s
 => => naming to ghcr.io/ucf-sttc/navsim/navsim                        0.0s
```

### Initial setup

  Modify the lines 6-8 of `navsim/docker-compose.yml` for paths specific to your system.

  * `$HOME/exp/` : Experiments are run in this folder
  * `$HOME/unity-envs/` : Unity-based standalone binaries are kept here
  * `/data`: This is where all the above symlinked folders are present. Please note in line 7 of `docker-compose.yml`: because on our systems, all our folders such as `workspaces/navsim`, `exp ` and `unity-envs` reside in `/data` and are symlinked in home folder, hence this `/data` folder also has to be mounted else the symlinks wont work in container. If you are not using symlinks, then you can remove this line.

### Test the container

  Run the following commands to test everything works fine:

  `docker compose run --rm navsim-test`

  `docker compose run --rm navsim navsim --help`

  In the following test command replace `ARORA/ARORA.x86_64` with the path to your unity binary that you mapped in `x-data: &data section` of docker-compose in above instructions. In our case it is the foldername and binary after `$HOME/unity-envs` that we mapped in `x-data: &data` section of docker-compose earlier: `ARORA/ARORA.x86_64`.

  `docker compose run --rm navsim navsim --plan --env arora-v0 --show_visual --env_path /unity-envs/ARORA/ARORA.x86_64`
  
### Run the experiments (inside container)

  Run the following command: (remove `-d` after `run` to run it in foreground):
  
  `docker compose run -d --rm navsim <navsim command>`

  [`<navsim command>`](#the-navsim-command-examples) is described in the section below.


## Run navsim in the host (without container)

### Install pre-requisites for using in the host (without container)

* Install `mamba 4.12.0-3` : `https://github.com/conda-forge/miniforge/releases/download/4.12.0-3/Mambaforge-4.12.0-3-Linux-x86_64.sh`
* ```sh
  cd /path/to/navsim/repo
  mamba create -n navsim python==3.8.16
  mamba env update -n navsim -f pyenv/navsim.yml
  conda activate navsim
  ./install-repo.sh
  ```
### Run the experiments (in the host, without container)

* Read `navsim_envs` tutorial to use and test the `navsim_envs`
* Run `jupyter notebook`. The notebooks are in `examples` folder.
* Run the [`<navsim command>`](#the-navsim-command-examples) is described in the section below

## The `<navsim command>` examples

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

## Errors FAQ

### vulkan error while starting the container
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

2. Update the id inside the container as per section: [fix the user id inside the container](#fix-the-user-id-inside-the-container)


## Contributing to NavSim API

### build and test flow

* Modify code


### release flow
* switch to feature branch
* Modify the `version.txt`
* Modify image version in:
  * docker-compose.yml
  * .github/workflows/deploy-docs.yml
* Build the container: `docker compose build navsim-build`
* Run fixid: `DUID=``id -u`` DGID=``id -g`` docker compose build navsim-fixid`
* Test the container
* Commit and push the changes
* create a pull request to main branch
* Merge the pull request
* Switch to main branch: `git checkout main`
* Build the container: `docker compose build navsim-build`
* Push the container: `docker compose push navsim-build`
* `git tag vx.x.x` and `git push --tags`
* Run the docs workflow in github manually
* Create a release in github with the tag

### General dev info:
* Use only google style to document your code:
  https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google

* `/opt/navsim` : Navsim code folder is kept here

### to modify and build the docs:

1. Create `<version>-docs` branch from `master` or `version` branch
2. Make sure you have navsim conda env activated and local repo installed with pip.
3. Modify the `.md` files in `/path/to/navsim`, `/path/to/navsim-lab`, `/path/to/navsim-envs`
4. go to `navsim/docs`
5. run `make html latexpdf`. If still errors, please call me and I will help fix
6. `pdf` and `html` versions are inside the `docs/build/...` folders
7. If you are happy with the PDF/formatting etc then commit and push the doc branch back

### misc tasks:
To give the zip of repo to someone, run the following command:

```sh
zip -FSr repo.zip \
    navsim-lab navsim-envs \
    navsim-mlagents/ml-agents-envs \
    navsim-mlagents/gym-unity \
    version.txt \
    install-repo.sh \
    examples \
    -x@.dockerignore
```
