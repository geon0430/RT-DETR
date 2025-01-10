FROM nvcr.io/nvidia/pytorch:23.01-py3

ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

COPY . /rt-detr-paddle-build-onnx-tensorrt

RUN bash /rt-detr-paddle-build-onnx-tensorrt/setting-scripts/install_dependencies.sh
RUN bash /rt-detr-paddle-build-onnx-tensorrt/setting-scripts/install_pip.sh

