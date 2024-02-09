#!/usr/bin/env bash

IMAGE=francescoruscelli/horizon
CONTAINER_NAME=borealis
docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix --name ${CONTAINER_NAME} --env=DISPLAY --network="host" ${IMAGE} 
#docker run -it --rm --privileged --net="host" --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all --env=DISPLAY --env=QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix --gpus all ${IMAGE} /bin/bash
#docker run -it --rm --privileged --net=host --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all --env=DISPLAY --env=QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 ${IMAGE} /bin/bash
