#!/bin/bash

port_num="1"
CONTAINER_NAME="paddle_RT-DETR_model_build_onnx"
IMAGE_NAME="paddle_RT-DETR_model_build_onnx"
TAG="0.1"

port_num="1"
RT-DETR-paddle-model-build-onnx_path=$(pwd)


docker run \
    --runtime nvidia \
    --gpus all \
    -it \
    -p ${port_num}8888:8888 \
    --name ${CONTAINER_NAME} \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ${RT-DETR-paddle-model-build-onnx_path}:/RT-DETR-paddle-model-build-onnx \
    -e DISPLAY=$DISPLAY \
    --shm-size 20g \
    --restart=always \
    -w /RT-DETR-paddle-model-build-onnx \
    ${IMAGE_NAME}:${TAG}
