# Contributing to NavSim API

## General dev info:
* Use only google style to document your code:
  https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google
  
## How to setup dev laptop to code for navsim API
* clone the ai_coop_py repo
  ```
  git clone <blah blah>
  ```
* Install miniconda (if not already installed)
  ```
  NAVSIM_ROOT=~/projects/ai_coop_py
  cd $NAVSIM_ROOT/navsim_env
  CONDA_ROOT=/opt/conda
  sudo mkdir $CONDA_ROOT
  sudo chown $(id -u) $CONDA_ROOT
  source ezai-conda.sh && install_miniconda
  exit
  ```
  
* Create the conda env for `navsim`
  
  ```
  NAVSIM_ROOT=~/projects/ai_coop_py
  cd $NAVSIM_ROOT/navsim_env
  ENVS_ROOT=/opt/conda/envs
  source ezai-conda.sh; ezai_conda_create --venv "$ENVS_ROOT/navsim"
  conda activate navsim
  cd $NAVSIM_ROOT
  pip install -e .
  ```
  
## Testing from local repo

For IST Devs: From local docker repo for development purposes:
```
repo="localhost:5000"
```
