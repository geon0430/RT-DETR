#!/bin/bash

IMAGE_NAME="rt-detr-paddle-model-build-onnx"
TAG="0.1"

docker build --no-cache -t ${IMAGE_NAME}:${TAG} .
