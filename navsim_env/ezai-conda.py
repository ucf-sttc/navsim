#!/usr/bin/env python3

from conda.cli.python_api import Commands, run_command

#venv='ezai'
def prefix():
  conda_base = run_command(Commands.INFO, " --base")
  venv = f"{conda_base}/envs/ezai"

  return

install_jupyter () {
  echo "Installing jupyter ..."
  conda install -y -S -c conda-forge "ipython>=7.0.0" "notebook>=6.0.0" jupyter_contrib_nbextensions jupyter_nbextensions_configurator yapf ipywidgets ipykernel ipympl
  return $?
}

install_jupyter_extensions () {
  echo "Setting jupyter extensions ..."
  conda install -y -S -c conda-forge jupyter_contrib_nbextensions jupyter_nbextensions_configurator yapf ipywidgets && \
  jupyter nbextension enable --sys-prefix code_prettify/code_prettify  && \
  jupyter nbextension enable --sys-prefix toc2/main && \
  jupyter nbextension enable --sys-prefix varInspector/main && \
  jupyter nbextension enable --sys-prefix execute_time/ExecuteTime && \
  jupyter nbextension enable --sys-prefix spellchecker/main && \
  jupyter nbextension enable --sys-prefix scratchpad/main && \
  jupyter nbextension enable --sys-prefix collapsible_headings/main && \
  jupyter nbextension enable --sys-prefix codefolding/main
  return $?
}

# sets passed prefix env as jupyter kernel
install_jupyter_kernel () {
  kernel_name='ezai'
  kernel_disp_name='ezai'
  conda install -y -S -c conda-forge ipykernel
  python -m ipykernel install --prefix="$1" --name $kernel_name --display-name $kernel_disp_name
  return $?
}

create_venv () {
  echo "$venv doesnt exist - creating now with python $py_ver ..."
  conda create -y -p $venv -c conda-forge "python=$py_ver" "conda=4.6.14" "pip=20.2.2"
  return $?
}

install_venv () {
  echo "installing python $py_ver in $venv..."
  conda install -y -S -c conda-forge "python=${py_ver}" "conda=4.6.14" "pip=20.2.2"
  return $?
}

config_env () {
  conda config --env --append channels conda-forge && \
  conda config --env --set auto_update_conda False && \
  #conda config --env --set channel_priority strict && \
  conda config --env --remove channels defaults
  return $?
}

#install_jupyter () {
#  echo "Installing jupyter ..."
#  conda install -y -S -c conda-forge "ipython>=7.0.0" "notebook>=6.0.0" jupyter_contrib_nbextensions jupyter_nbextensions_configurator yapf ipywidgets ipykernel && \
#  jupyter nbextension enable --user code_prettify/code_prettify  && \
#  jupyter nbextension enable --user toc2/main && \
#  jupyter nbextension enable --user varInspector/main && \
#  jupyter nbextension enable --user execute_time/ExecuteTime && \
#  jupyter nbextension enable --user spellchecker/main && \
#  jupyter nbextension enable --user scratchpad/main && \
#  jupyter nbextension enable --user collapsible_headings/main && \
#  jupyter nbextension enable --user codefolding/main && \
#  return $?
#}

install_cuda () {
  echo "Installing cuda ..."
  conda install -y -S -c conda-forge -c defaults "cudatoolkit=10.1" "cudnn>=7.6.5" && \
  conda install -y -S "nccl" #"mpi4py>=3.0.0" gxx_linux-64 gcc_linux-64
  return $?
}

install_fastai_pytorch () {
  echo "Installing fastai and pytorch ..."
  conda config --env --prepend channels pytorch
  conda config --env --prepend channels fastai
  conda config --show-sources
  # numpy spec due to tensorflow and pillow spec due to gym
  conda install -y -S "fastai=2.0.0" "pytorch=1.6.0" "torchvision=0.7.0" "numpy<1.19.0" #"gym=0.18.0"
  return $?
}

install_txt () {
  conda config --show-sources
  conda install -y -S --file $condatxt && \
  # install pip with no-deps so it doesnt mess up conda installed versions
  pip install --no-deps --no-cache-dir -r "$piptxt"
  return $?
}

ezai_conda_create () {
  venv=${venv:-$(conda info --base)/envs/ezai}
  piptxt=${piptxt:-"./ezai-pip-req.txt"}
  condatxt=${condatxt:-"./ezai-conda-req.txt"}
  py_ver=${py_ver:-3.7.3}
  # add -k if ssl_verify needs to be set to false
  
  while [ $# -gt 0 ]; do
     if [[ $1 == *"--"* ]]; then
          param="${1/--/}"
          declare $param="$2"
          #echo $1 $2 #// Optional to see the parameter:value result
     fi
    shift
  done

  conda clean -i
  source $(conda info --base)/etc/profile.d/conda.sh

  if [ "${venv}" != "base" ];
  then
    echo "setting base conda to 4.6.14, python to 3.7.3"
    activate base
    conda config --env --set auto_update_conda False
    conda config --show-sources
    conda install -y -S --no-update-deps "conda=4.6.14" "python=3.7.3" || (echo "Unable to update base conda"; exit 1)
    deactivate

    activate "${venv}" || create_venv || (echo "Unable to create ${venv}" ; exit 1)
  else
    activate "${venv}" && install_venv
    deactivate
  fi

  activate "${venv}" && config_env
  deactivate

  activate "${venv}" && ( install_cuda && install_fastai_pytorch && install_txt )
  deactivate

  # Expose environment as kernel
  #python -m ipykernel install --user --name ezai-conda --display-name "ezai-conda"

  # TODO: Uncomment below in final version
  if [ "${venv}" != "base" ];
  then
    activate "${venv}" &&  conda clean -yt
    deactivate
  fi
  activate base && conda clean -yt
  deactivate
  find $(conda info --base) -follow -type f -name '*.a' -delete
  find $(conda info --base) -follow -type f -name '*.pyc' -delete

  # TODO: Uncomment above in final version
  echo " "
  echo " "
  echo " For Linux 64, Open MPI is built with CUDA awareness but this support is disabled by default."
  echo "To enable it, please set the environmental variable OMPI_MCA_opal_cuda_support=true before"
  echo "launching your MPI processes. Equivalently, you can set the MCA parameter in the command line:"
  echo "mpiexec --mca opal_cuda_support 1 ..."

  echo " "
  echo " "
  echo "Activate your environment with  conda activate $venv  and then test with pytest -p no:warnings -vv"
}



