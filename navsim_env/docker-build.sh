#!/usr/bin/env bash

export local=0
export sing=1
export itag="1.0.0"

options=$(getopt -l "remote,singularity" -a -o "rs" -- "$@")
eval set -- "$options"

while true; do
  case $1 in
  -r | --remote)
    local="1"
    ;;
  -s | --singularity)
    sing="0"
    ;;
  --)
    shift
    break
    ;;
  esac
  shift
done

irepo="ghcr.io/armando-fandango" #image repo
iname="${irepo}/navsim:${itag}"  #image name
lrepo="localhost:5000"
lname="${lrepo}/navsim:${itag}"

#iname="ghcr.io/armando-fandango/navsim:0.0.1"   #image name
dfile="Dockerfile"
dfolder="."

image_exists() {
  echo parameter=$1
  docker image inspect $1 >/dev/null 2>&1
  return $?
}

# "$(docker image inspect $iname > /dev/null 2>&1 && echo 1 || echo '')"
if [ "$(image_exists $iname)" ]; then
  echo "found image $iname.. rebuilding"
else
  echo "creating image $iname"
fi

bopt=" "
bopt+=" --no-cache "
docker build $bopt -t $iname -f $dfile $dfolder

# HAVE TO CHECK FOR IMAGE AGAIN BECAUSE BUILD FAILS SOMETIME
if [ "$(image_exists $iname)" ]; then
  echo "created image $iname"

  if [[ $local ]]; then
    docker tag $iname $lname
    docker push $lname
  else
    echo "no $local"
    #docker push $iname
  fi
else
  echo "not created image $iname"
fi

#
# docker run -d -p 5000:5000 --restart=always --name registry -v /mnt/data/docker-registry:/var/lib/registry registry:2
# docker container stop registry && docker container rm -v registry

# ver=0.0.2
# singularity pull docker://ghcr.io/armando-fandango/navsim:$ver
# SINGULARITY_NOHTTPS=true singularity pull docker://localhost:5000/navsim:$ver
# singularity shell --bind /mnt --nv navsim_${ver}.sif
# docker run -it --gpus all --name navsim_${ver}_1 \
#      -u $(id -u):$(id -g) -w ${PWD} \
#      -v /mnt:/mnt
#      -e DISPLAY -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all \
#      navsim_${ver} bash
