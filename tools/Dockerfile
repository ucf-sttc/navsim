FROM nvidia/cudagl:11.1.1-devel-ubuntu20.04 as base-image
MAINTAINER Armando Fandango <armando@neurasights.com>

#ARG COMPONENTS=Unity,Windows,Windows-Mono,Mac,Mac-Mono,WebGL
#ARG COMPONENTS=Unity,Mac,WebGL

ENV CONDA_DIR="/opt/conda"
ENV DEBIAN_FRONTEND=noninteractive

USER root
RUN apt-get -qq update && \
    apt-get -qq upgrade -y && \
    apt-get -qq install -y --no-install-recommends \
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
	  gdebi-core \
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
      kmod \
      # for phoronix
      php-cli \
      php-xml \
      pciutils && \
    apt-get autoremove --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

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

#install phoronix
RUN wget http://phoronix-test-suite.com/releases/repo/pts.debian/files/phoronix-test-suite_10.4.0_all.deb && \
    gdebi phoronix-test-suite_10.4.0_all.deb && \
    rm phoronix-test-suite_10.4.0_all.deb
#RUN dpkg -i phoronix-test-suite_10.4.0_all.deb
COPY test-profiles/local /var/lib/phoronix-test-suite/test-profiles/

#Install MINICONDA and navsim_env in base
COPY scripts/* ezai-conda* ezai-pip-* repo.zip /root/

#COPY repo.zip /root/
#RUN ls -l /root/navsim-repo/navsim-mlagents/
#RUN read -p "Press any key to resume ..."
#COPY mlagents_patch /root/mlagents_patch
RUN chmod +x /root/*.sh && \
    source ezai-conda.sh && \
    install_miniconda --conda_dir ${CONDA_DIR}
ENV PATH=${CONDA_DIR}/bin:$PATH
#ENV BASH_ENV="/root/.bashrc"
RUN unzip -d ${CONDA_DIR}/navsim-repo repo.zip && \
    source /root/.bashrc && source /root/ezai-conda.sh && \
    ezai_conda_create --venv "base" && \
    activate base && \
    pip install -e ${CONDA_DIR}/navsim-repo/navsim-mlagents/ml-agents-envs && \
    pip install -e ${CONDA_DIR}/navsim-repo/navsim-mlagents/gym-unity && \
    pip install -e ${CONDA_DIR}/navsim-repo/navsim-envs && \
    pip install -e ${CONDA_DIR}/navsim-repo/navsim-lab && \
    chmod -R 777 ${CONDA_DIR} && \
    rm ezai-conda-* ezai-pip-* repo.zip
#ENTRYPOINT ["bash", "-c", "(/root/chuser.sh && python /root/x_server.py) && bash"]
ENTRYPOINT ["/root/entrypoint.sh"]
CMD ["bash"]