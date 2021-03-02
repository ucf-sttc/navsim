#!/usr/bin/env bash

# to pull: docker pull <image>
# to remove image: docker rmi <image>
# to remove container: docker stop <container> && docker rm <container>

#Flags:
#--xserver: starts container with xserver running

itag=${itag:-"1.0.0-dev-headless"}
cname=${cname:-'navsim-1.0.0-dev-headless-1'}
while [ $# -gt 0 ]; do
   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        echo $1 $2 #// Optional to see the parameter:value result
   fi
  shift
done

irepo="ghcr.io/armando-fandango" #image repo
iname="${irepo}/navsim:${itag}"   #image name

#
# docker run -it --gpus all --name phd-gpu-1 -v ${HOME}/datasets:/root/datasets -v /home/armando/phd:/root/phd

# test cvfb : xvfb-run -s "-screen 0 1024x768x24" glxgears
# ports exposed in format -p host-port:container-port
#cports=" -p 8888:8888 " # jupyter notebook
#cports+=" -p 6006:6006 " # tensorboard
#cports+=" -p 4040:4040 " # spark webui
#cports+=" -p 5004:5004 " # unity
#cports+=" -p 5005:5005 " # unity

wfolder=" -w ${PWD}"
vfolders=" "
vfolders+=" -v ${HOME}:${HOME}"
vfolders+=" -v /mnt:/mnt "
vfolders+=" -v /data:/mnt/work "
vfolders+=" -v /tmp/.X11-unix:/tmp/.X11-unix "
           #-v ${HOME}/.local/share/unity3d /root/.local/share/unity3d "
           #-v ${HOME}/.Xauthority:/root/.Xauthority:rw "
vfolders+=" -v /etc/group:/etc/group:ro"
vfolders+=" -v /etc/passwd:/etc/passwd:ro"
vfolders+=" -v /etc/shadow:/etc/shadow:ro"

#dfile="Dockerfile"
#dfolder="."

# exec options
evars=" -e DISPLAY -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all "
user=" -u $(id -u):$(id -g)"
cmd="/root/startx.sh; bash"
xhost +  # for running GUI app in container

if [ "$(docker image inspect $iname > /dev/null 2>&1 && echo 1 || echo '')" ];
then
  echo "found image $iname"
else
  docker pull $iname
fi

# HAVE TO CHECK FOR IMAGE AGAIN BECAUSE BULD/PULL FAILS SOMETIME
if [ "$(docker image inspect $iname > /dev/null 2>&1 && echo 1 || echo '')" ];
then
  if [ "$(docker container inspect $cname > /dev/null 2>&1 && echo 1 || echo '')" ];
  then
    echo "found container $cname"
    if [ "$(docker container inspect $cname -f '{{.State.Status}}')"!="running" ]
    then
      echo "starting container $cname"
      docker start "$cname"
    fi
    echo "entering started container $cname"
    echo "docker exec -it ${evars} $cname bash"
    #TODO: refine privileged status to individual devices
    #defaulted to root user to allow ability to run X server
    #TODO: add individual user login
	#Leaving the docker container open means Xserver is on and taking up memory
	#Could lead to memory leak
    docker exec --privileged -it ${evars} $cname bash
  else
    echo "creating, starting and then entering container $cname"
    # shellcheck disable=SC2154
	docker run --privileged -it --gpus all --name $cname \
	$evars $vfolders $cports $iname bash
  fi
else
   echo "image $iname not found"
fi

#nvidia-xconfig -o headlessxorg.conf -a --use-display-device=None --virtual=1280x1024
