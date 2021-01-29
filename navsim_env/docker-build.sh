#!/usr/bin/env bash

export local=1
export sing=1

options=$(getopt -l "local,singularity" -a -o "ls" -- "$@")
eval set -- "$options"

while true
do
  case $1 in
    -l|--local)
      local="0"
      ;;
    -s|--singularity)
      sing="1"
      ;;
    --)
      shift
      break;;
  esac
  shift
done

itag="0.0.1"
irepo="ghcr.io/armando-fandango" #image repo
iname="${irepo}/navsim:${itag}"   #image name
lrepo="localhost:5000"
lname="${lrepo}/navsim:${itag}"

#iname="ghcr.io/armando-fandango/navsim:0.0.1"   #image name
dfile="Dockerfile"
dfolder="."

image_exists () {
  echo parameter=$1
  docker image inspect $1 > /dev/null 2>&1
  return $?
}

# "$(docker image inspect $iname > /dev/null 2>&1 && echo 1 || echo '')"
if [ $(image_exists $iname) ];
then
  echo "found image $iname.. rebuilding"
else
  echo "creating image $iname"
fi

bopt=" "
bopt+=" --no-cache "
docker build $bopt -t $iname -f $dfile $dfolder

# HAVE TO CHECK FOR IMAGE AGAIN BECAUSE BUILD FAILS SOMETIME
if [ image_exists $iname ];
then
   echo "created image $iname"
   #docker push $iname
else
   echo "not created image $iname"
fi

if [ $local ];
then
  docker tag $iname $lname
  docker push $lname
else
  echo "no $local"
fi

if [ $sing ];
then
  [ $local ] && (singularity pull lname) || (singularity pull iname)
fi
# docker run -d -p 5000:5000 --restart=always --name registry registry:2
# docker container stop registry && docker container rm -v registry
# docker tag $iname $lname
# docker push $lname

# singularity pull docker://ghcr.io/armando-fandango/navsim:0.0.1'
# singularity shell --bind /mnt --nv navsim_0.0.1.sif'