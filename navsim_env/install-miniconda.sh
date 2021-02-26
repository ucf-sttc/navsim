#!/usr/bin/env bash

# run this with sudo

conda_dir=${conda_dir:-/opt/conda}

while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
   fi
  shift
done

wget -nv https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh -O Miniconda.sh && \
#wget -nv https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O Miniconda.sh && \
#curl -o Miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh && \
	/bin/bash Miniconda.sh -f -b -p ${conda_dir} && \
	rm Miniconda.sh && \
  PATH=${conda_dir}/bin:$PATH && \
  #source $(conda info --base)/etc/profile.d/conda.sh && \
  # shellcheck disable=SC2046
  conda init $(basename $SHELL)
  #&& \
  #chmod -R 777 ${conda_dir}
#  chown -R `id -u`:`id -g` ${HOME}/.conda
