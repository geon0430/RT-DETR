FROM nvcr.io/nvidia/pytorch:23.01-py3

ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

ENV XDG_RUNTIME_DIR "/tmp"

RUN python3 -m pip install --upgrade pip

WORKDIR /
RUN mkdir -p /RT-DETR-paddle-model-build-onnx
COPY . /RT-DETR-paddle-model-build-onnx

RUN bash /python_object_detection_server/setting-scripts/install_dependencies.sh

RUN pip install pandas python-multipart paddlepaddle-gpu paddlepaddle paddle2onnx onnxruntime

RUN bash /python_object_detection_server/setting-scripts/install_OpenCV.sh

