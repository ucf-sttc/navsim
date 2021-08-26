# Contributing to NavSim API

## General dev info:
* Use only google style to document your code:
  https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google
  
## How to setup dev laptop to code for navsim API
* clone the ai_coop_py repo and checkout the active branch
  ```
  git clone git@ssh.dev.azure.com:v3/emb-ist-edu/AI%20COOP/ai_coop_py
  git submodule update --init
  git checkout 2.10.9
  ```
* Follow instructions in setting up navsim conda environment on the host. Activate the navsim conda environment.
* Either setup `navsim` conda env or with : 
  `conda create -n navsim python=3.8 jupyter navsim_lab navsim_envs && conda activate navsim` 
* `pip install -e /path/to/ai_coop_py/navsim-envs`
* `pip install -e /path/to/ai_coop_py/navsim-lab`
* Run `jupyter notebook`. The notebooks are in examples folder.

## to modify and build the docs:

1. Create `<version>-docs` branch from `master` or `version` branch
2. Make sure you have navsim conda env activated and local repo installed with pip.
3. Modify the `.md` files in `/path/to/navsim-lab`, `/path/to/ai_coop_py`, `/path/to/navsim-envs`
4. go to `ai_coop_py/docs`
5. run `make html latexpdf`. If it results in error, just run `make latexpdf html`. If still errors, please call me and I will help fix
6. The pdf and html docs would be dumped in the `ai_coop_py/docs` from where you ran the make.
7. If you are happy with the PDF/formatting etc then commit and push the doc branch back

## Testing from local repo

For IST Devs: From local docker repo for development purposes:
```
repo="localhost:5000"
```
