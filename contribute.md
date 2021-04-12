# How to setup dev laptop to code for navsim API
* clone the ai_coop_py repo
  ```
  git clone <blah blah>
  ```
* Install miniconda and the env
  ```
  NAVSIM_ROOT=~/projects/ai_coop_py
  cd $NAVSIM_ROOT/navsim_env
  sudo mkdir /opt/conda
  sudo chown $(id -u) /opt/conda
  source ezai-conda.sh && install-miniconda
  exit
  
  NAVSIM_ROOT=~/projects/ai_coop_py
  cd $NAVSIM_ROOT/navsim_env
  rm -rf /opt/conda/envs/navsim; source ezai-conda.sh; ezai_conda_create --venv "/opt/conda/envs/navsim"
  conda activate navsim
  cd $NAVSIM_ROOT
  pip install -e .
  ```
  
# Testing from local repo

For IST Devs: From local docker repo for development purposes:
```
repo="localhost:5000"
```
