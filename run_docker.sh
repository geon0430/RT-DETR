#!/bin/bash

port_num="1"
CONTAINER_NAME="paddle_RT-DETR_model_build_onnx"
IMAGE_NAME="paddle_RT-DETR_model_build_onnx"
TAG="0.1"

port_num="1"
fastapiUiUx_path=$(pwd)


docker run \
    --runtime nvidia \
    --gpus all \
    -it \
    -p ${port_num}1575:1554 \
    -p ${port_num}6000:8000 \
    -p ${port_num}7000:9000 \
    -p ${port_num}7777:8888 \
    -p ${port_num}7444:8444 \
    --name ${CONTAINER_NAME} \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ${fastapiUiUx_path}:/RT-DETR \
    -e DISPLAY=$DISPLAY \
    --shm-size 20g \
    --restart=always \
    -w /RT-DETR \
    ${IMAGE_NAME}:${TAG}
