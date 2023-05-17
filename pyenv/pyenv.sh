#!/usr/bin/env bash

function ctrl_c() {
        echo "Requested to stop."
        trap '' INT
        exit 1
}


activate () {
  conda activate $1 || source activate $1
}

deactivate () {
  conda deactivate || source deactivate
}

# sets passed prefix env as jupyter kernel in currently active env
install_jupyter_kernel () {
  venv=${venv:-'ezai'}
  kernel=
  #conda install -y -S -c conda-forge ipykernel

  python -m ipykernel install --sys-prefix --prefix="$1" --name $venv --display-name $venv
  return $?
}

install_jupyter () {
  echo "Installing jupyter ..."
  mamba env update --file jupyter.yml
  return $?
}

config_jupyter () {
  echo "Configuring jupyter..."
  jupyter labextension enable @jupyterlab/debugger && \
  jupyter labextension enable @jupyterlab/toc && \
  jupyter labextension enable @jupyterlab/execute_time && \
  jupyter labextension enable @jupyterlab/nvdashboard && \
  jupyter contrib nbextension install --sys-prefix && \
  jupyter nbextension enable --sys-prefix code_prettify/code_prettify  && \
  jupyter nbextension enable --sys-prefix toc2/main && \
  jupyter nbextension enable --sys-prefix varInspector/main && \
  jupyter nbextension enable --sys-prefix execute_time/ExecuteTime && \
  jupyter nbextension enable --sys-prefix spellchecker/main && \
  jupyter nbextension enable --sys-prefix scratchpad/main && \
  jupyter nbextension enable --sys-prefix collapsible_headings/main && \
  jupyter nbextension enable --sys-prefix codefolding/main && \
  jupyter nbextension enable --sys-prefix jupyter_resource_usage/main
  return $?
}

install_py () {
  echo "installing python $py_ver"
  mamba install -y -S -c conda-forge "python=$py_ver"
  return $?
}

create_venv () {
  echo "$venv doesnt exist - creating now with python $py_ver ..."
  mamba create -y -n $venv -c conda-forge "python=$py_ver"
  return $?
}

config_venv () {
  #conda config --env --append channels conda-forge && \
  #conda config --env --set auto_update_conda False && \
  #conda config --env --set channel_priority strict && \
  #conda config --env --remove channels defaults
  mamba env config vars set MAMBA_NO_BANNER=1
  return $?
}

install_cuda () {
  echo "Installing cuda ..."
  mamba env update --file cuda.yml
  return $?
}

install_rapids () {
  echo "Installing RAPIDS libraries ..."
  conda config --env --prepend channels rapidsai
  conda config --show-sources
  conda install -y -S rapids
  return $?
}

install_pytorch () {
  echo "Installing pytorch ..."
  mamba env update --file pytorch.yml
  return $?
}

install_ros () {
  echo "Installing robostack ROS ..."
  conda config --env --prepend channels robostack
  conda config --show-sources
  mamba install ros-noetic-desktop catkin_tools rosdep jupyter-ros jupyterlab-ros gazebo ros-noetic-turtlebot3
  return $?
}

install_habitat() {
  #conda config --env --prepend channels aihabitat
  #conda config --show-sources
  #conda install -c conda-forge -c aihabitat "habitat-sim=0.1.7" withbullet headless
  #TODO: Does this need to be in the Dockerfile-ubuntu2004 ?

  git clone https://github.com/facebookresearch/habitat-sim.git /opt/habitat-sim && \
    cd /opt/habitat-sim && \
    git checkout 9575dcd45fe6f55d2a44043833af08972a7895a9 && \
    pip install -r /opt/habitat-sim/requirements.txt && \
    python setup.py install --headless && \
    cd - && \
    git clone https://github.com/facebookresearch/habitat-lab.git /opt/habitat-lab && \
    cd /opt/habitat-lab && \
    git checkout b5f2b00a25627ecb52b43b13ea96b05998d9a121 && \
    pip install -e /opt/habitat-lab && \
    wget http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip && \
    unzip habitat-test-scenes.zip && \
    cd - && \
    chmod -R 777 ${CONDA_DIR} /opt/habitat*
    return $?
}

install_detectron() {
  # Install detectron2
  pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
}

install_yml () {
  echo "Installing conda packages from yml file ..."
  mamba env update --file misc.yml
  # install pip with no-deps so it doesnt mess up conda installed versions
  # pip install --no-cache-dir -r "$piptxt" --no-deps
  return $?
}

ez_create_env () {
  venv=${venv:-$(mamba info --base)/envs/ezai}
  piptxt=${piptxt:-"./ez-pip-req.txt"}
  py_ver=${py_ver:-3.8}

  trap ctrl_c INT
  
  # add -k if ssl_verify needs to be set to false
  
  while [ $# -gt 0 ]; do
     if [[ $1 == *"--"* ]]; then
          param="${1/--/}"
          declare $param="$2"
          #echo $1 $2 #// Optional to see the parameter:value result
     fi
    shift
  done

  export MAMBA_NO_BANNER=1
  mamba env config vars set MAMBA_NO_BANNER=1
  mamba clean -i
  #source $(conda info --base)/etc/profile.d/conda.sh

  if [ "${venv}" != "base" ];
  then
    #echo "no more setting base conda to 4.6.14, python to 3.7.3"
    #activate base
    #mamba config --env --set auto_update_conda False
    #conda config --show-sources
    #mamba env config vars set MAMBA_NO_BANNER=1
    #conda install -y --no-update-deps "conda=4.6.14" "python=3.7.3" || (echo "Unable to update base conda"; exit 1)
    #deactivate
    activate "${venv}" || create_venv || (echo "Unable to create ${venv}" && return $?)
    activate "${venv}"
  else
    activate "${venv}" && install_py
  fi

  config_venv

  install_cuda && install_pytorch && install_yml
  install_result="$?"
  if [ "$install_result" != "0" ];
  then
    echo "Conda install failed in ${venv}" && return `echo $install_result`
  fi
  # Expose environment as kernel
  #python -m ipykernel install --user --name ezai-conda --display-name "ezai-conda"

  # TODO: Uncomment below in final version
  #mamba clean -yt
  deactivate
  #find $(conda info --base) -follow -type f -name '*.a' -delete
  #find $(conda info --base) -follow -type f -name '*.pyc' -delete
  # TODO: Uncomment above in final version
  echo " "
  echo " "
  echo "Activate your environment with  mamba activate $venv"
  trap '' INT
}

ez_install_mamba () {
  mamba_dir=${mamba_dir:-/opt/py}
  #sudo mkdir $mamba_dir; sudo chown -R $USER $mamba_dir
  trap ctrl_c INT
  while [ $# -gt 0 ]; do
     if [[ $1 == *"--"* ]]; then
          param="${1/--/}"
          declare "$param"="$2"
          # echo $1 $2 // Optional to see the parameter:value result
     fi
    shift
  done

  wget -nv https://github.com/conda-forge/miniforge/releases/download/4.12.0-3/Mambaforge-4.12.0-3-Linux-x86_64.sh -O mamba.sh && \
  #curl -o Miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh && \
    /bin/bash mamba.sh -f -b -p "${mamba_dir}" && \
    PATH=${mamba_dir}/bin:$PATH && \
    #source $(conda info --base)/etc/profile.d/conda.sh && \
    mamba env config vars set MAMBA_NO_BANNER=1 && export MAMBA_NO_BANNER=1 && \
    mamba init $(basename "${SHELL}") && \
    which mamba && mamba -V && which python && python -V
    retval=$?
  rm mamba.sh
    #&& \
    #chmod -R 777 ${conda_dir}
  #  chown -R `id -u`:`id -g` ${HOME}/.conda
  trap '' INT
  return $retval
}

ez_install_node () {
    node_dir=${node_dir:-/opt/node}

    while [ $# -gt 0 ]; do
        if [[ $1 == *"--"* ]]; then
            param="${1/--/}"
            declare "$param"="$2"
            # echo $1 $2 // Optional to see the parameter:value result
        fi
        shift
    done

    cd ${node_dir}
    wget -nv https://nodejs.org/dist/v16.14.0/node-v16.14.0-linux-x64.tar.xz -O - | tar --strip-components=1 -xJv && \
    node -v && \
    npm -v && \
    npx -v

    return $?
}
