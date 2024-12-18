#!/bin/bash

port_num="1"
CONTAINER_NAME="rt-detr-paddle-model-build-onnx"
IMAGE_NAME="rt-detr-paddle-model-build-onnx"
TAG="0.1"
code_path=$(pwd)

docker run \
    --runtime nvidia \
    --gpus all \
    -it \
    -p ${port_num}8888:8888 \
    --name ${CONTAINER_NAME} \
    --privileged \
    -v ${code_path}:/rt-detr-paddle-build-onnx-tensorrt \
    --shm-size 5g \
    --restart=always \
    -w /rt-detr-paddle-build-onnx-tensorrt \
    ${IMAGE_NAME}:${TAG}
