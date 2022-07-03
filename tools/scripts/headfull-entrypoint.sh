#!/usr/bin/env bash

# Add local user
# Either use the LOCAL_USER_ID if passed in at runtime or
# fallback
DUID=${DUID:-0}
#NAVSIM_DIR="/opt/navsim"
#CONTAINER_ALREADY_STARTED="/tmp/CONTAINER_ALREADY_STARTED_PLACEHOLDER"
#if [ ! -e $CONTAINER_ALREADY_STARTED ]; then
#    touch $CONTAINER_ALREADY_STARTED
#    echo "-- First container startup --"
#    # YOUR_JUST_ONCE_LOGIC_HERE
#    echo "Installing nvidia drivers for NVIDIA_VERSION=${NVIDIA_VERSION}...."
#    NVIDIA_VERSION=$NVIDIA_VERSION /opt/container-scripts/install_nvidia.sh
    #echo "Upgrading navsim*......."
    #pip install --no-cache-dir --pre --upgrade "navsim>=2.10,<2.11"
    #pip install --no-cache-dir --upgrade  "navsim>=2.10,<2.11" "navsim_envs>=2.10,<2.11"
    #source /root/ezai-conda.sh && \
        #activate base && \
        #pip install -e /opt/conda/navsim-repo/navsim-mlagents/ml-agents-envs && \
        #pip install -e /opt/conda/navsim-repo/navsim-mlagents/gym-unity && \
        #pip install -e /opt/conda/navsim-repo/navsim-envs && \
        #pip install -e /opt/conda/navsim-repo/navsim-lab
#else
#    echo "-- Second or later container startup --"
#fi

nohup python /opt/container-scripts/x_server.py &
touch nohup.out
chmod 666 nohup.out
chown $DUID nohup.out
export HOME=${DUHOME:-/root}
echo "Starting with UID : $DUID"
exec chroot --skip-chdir --userspec=$DUID / bash -c "$*"