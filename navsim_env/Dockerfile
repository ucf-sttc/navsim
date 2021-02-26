FROM nvidia/cudagl:10.1-devel-ubuntu18.04 as base-image

#ARG COMPONENTS=Unity,Windows,Windows-Mono,Mac,Mac-Mono,WebGL
#ARG COMPONENTS=Unity,Mac,WebGL

ENV CONDA_DIR="/opt/conda"
ENV DEBIAN_FRONTEND=noninteractive

USER root
RUN apt-get -qq update && \
    apt-get -qq upgrade -y && \
    apt-get -qq install -y --no-install-recommends\
    # TODO: sort these package names by alpha - docker BP
      # needed to install custom / launchpad repos
      software-properties-common \
      sudo \
      ca-certificates \
      run-one \
      build-essential \
      pkg-config \
      patchelf \
      cmake \
      # avoid curl - use wget
	  wget \
	  git \
	  bzip2 \
	  unzip \
	  zip \
	  htop \
	  tmux \
	  nano \
	  vim \
	  # for webdriver - folium
	  firefox-geckodriver \
	  # for matplotlib
	  libgl1-mesa-glx \
	  # for unity
	  mesa-utils \
	  libglu1-mesa-dev \
	  libgles2-mesa-dev \
	  libgl1-mesa-dev \
	  libegl1-mesa-dev \
	  libosmesa6-dev \
	  libglvnd-dev \
	  libjpeg-dev \
	  libpng-dev \
	  libglm-dev \
	  libglfw3-dev \
	  libx11-dev \
	  freeglut3-dev \
	  xorg-dev \
	  libomp-dev \
	  swig \
	  libsm6 \
	  libxext6 \
	  libxrender-dev \
	  xserver-xorg-core \
	  xorg \
	  xserver-xorg-video-dummy \
	  x11vnc \
	  ffmpeg \
         kmod 

COPY scripts/install_nvidia.sh scripts/startx.sh /root/
RUN NVIDIA_VERSION=$NVIDIA_VERSION /root/install_nvidia.sh

RUN apt-get autoremove --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

	#swig \
    # sumo pre-req
	#openjdk-8-jre-headless ca-certificates-java
	# unity pre-req
	#libcurl4-openssl-dev \
	#libgtk2.0-0 \
	#libsoup2.4-1 \
	#libarchive13 \
	#libpng16-16 \
	#libgconf-2-4 \
	#lib32stdc++6 \
	#libcanberra-gtk-module

WORKDIR /root/
SHELL ["/bin/bash", "-c"]

#Install MINICONDA and navsim_env in base
COPY ezai-conda* ezai-pip-* /root/
RUN source ezai-conda.sh && \
    install_miniconda --conda_dir ${CONDA_DIR}
ENV PATH=${CONDA_DIR}/bin:$PATH
#ENV BASH_ENV="/root/.bashrc"
RUN source /root/.bashrc && source ezai-conda.sh && \
    ezai_conda_create --venv "base" && \
    chmod -R 777 ${CONDA_DIR} && \
    rm ezai-conda* ezai-pip-*

#COPY conda-ez-update* /root/
#RUN ./conda-ez-update.sh && \
#    rm conda-ez-update*

    #&& \
    #echo "alias eznb='conda activate ezai-conda && jupyter notebook --ip=* --no-browser'" >> ${AI_HOME}/.bashrc &&\
    #echo "conda activate ezai-conda" >> ${AI_HOME}/.bashrc
                    #--allow-root
# TODO: Make RUN commands use the new environment:
# SHELL ["conda", "run", "-n", "ezai", "/bin/bash", "-c"]
# SHELL ["/bin/bash", "-c"] this doesnt work

