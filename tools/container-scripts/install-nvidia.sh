#!/bin/bash
#https://github.com/allenai/ai2thor-docker/blob/master/scripts/install_nvidia.sh

NVIDIA_INSTALLED="/tmp/NVIDIA_INSTALLED"
if [ ! -e $NVIDIA_INSTALLED ]; then
    touch $NVIDIA_INSTALLED
    # YOUR_JUST_ONCE_LOGIC_HERE
    nvidia_version=`cat /proc/driver/nvidia/version |grep 'NVRM version:'| grep -oE "Kernel Module\s+[0-9.]+"| awk {'print $3'}`
    echo "Installing nvidia drivers for NVIDIA_VERSION=${nvidia_version}...."
    nvidia_major_version=`echo $nvidia_version |sed "s/\..*//"`
    driver_filename="NVIDIA-Linux-x86_64-$nvidia_version.run"
    driver_url="http://us.download.nvidia.com/XFree86/Linux-x86_64/$nvidia_version/$driver_filename"
    tesla_driver_url="http://us.download.nvidia.com/tesla/$nvidia_version/$driver_filename"
    s3_driver_url="http://ai2-vision-nvidia.s3-us-west-2.amazonaws.com/$driver_filename"
    wget $driver_url -P /root/ || wget $tesla_driver_url -P /root/ || wget $s3_driver_url -P /root/

    if [ $?  -eq 0 ]; then
        sh /root/$driver_filename -s --no-kernel-module
    else
        echo "Error trying to install nvidia driver for $nvidia_version"
        exit 1
    fi

else
    echo "nvidia drivers already installed ... skipping."
fi


