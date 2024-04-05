#!/bin/bash

IMAGE_NAME="paddle_RT-DETR_model_build_onnx"
TAG="0.1"

docker build --no-cache -t ${IMAGE_NAME}:${TAG} .
