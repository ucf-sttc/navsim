#!/usr/bin/env bash

# Add local user
# Either use the LOCAL_USER_ID if passed in at runtime or
# fallback
USER_ID=${USER_ID:-0}
CONTAINER_ALREADY_STARTED="/tmp/CONTAINER_ALREADY_STARTED_PLACEHOLDER"
if [ ! -e $CONTAINER_ALREADY_STARTED ]; then
    touch $CONTAINER_ALREADY_STARTED
    echo "-- First container startup --"
    # YOUR_JUST_ONCE_LOGIC_HERE
    NVIDIA_VERSION=$NVIDIA_VERSION /root/install_nvidia.sh
    pip install --no-cache-dir --upgrade "navsim>=2.9,<2.10"
else
    echo "-- Second or later container startup --"
fi

nohup python /root/x_server.py &
touch nohup.out
chmod 666 nohup.out
chown $USER_ID nohup.out
export HOME=${USER_HOME:-/root}
echo "Starting with UID : $USER_ID"
exec chroot --skip-chdir --userspec=$USER_ID / bash -c "$*"