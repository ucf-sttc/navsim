#!/usr/bin/env bash

#nvidia-xconfig -o xcontainer.conf -a
# nvidia-xconfig -o xcontainer.conf -a --use-display-device=None --virtual=1280x1024
#X -config xcontainer.conf :88 &

CONTAINER_ALREADY_STARTED="CONTAINER_ALREADY_STARTED_PLACEHOLDER"
if [ ! -e $CONTAINER_ALREADY_STARTED ]; then
    touch $CONTAINER_ALREADY_STARTED
    echo "-- First container startup --"
    # YOUR_JUST_ONCE_LOGIC_HERE
    NVIDIA_VERSION=$NVIDIA_VERSION /root/install_nvidia.sh
else
    echo "-- Not first container startup --"
fi