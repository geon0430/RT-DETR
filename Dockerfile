FROM nvcr.io/nvidia/pytorch:23.01-py3

ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

COPY . /RT-DETR-paddle-model-build-onnx

RUN bash /RT-DETR-paddle-model-build-onnx/setting-scripts/install_dependencies.sh
RUN bash /RT-DETR-paddle-model-build-onnx/setting-scripts/install_pip.sh
RUN bash /RT-DETR-paddle-model-build-onnx/setting-scripts/install_OpenCV.sh

