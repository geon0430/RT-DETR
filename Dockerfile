FROM nvcr.io/nvidia/pytorch:23.01-py3

ENV TZ=Asia/Seoul
ENV DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all

ENV XDG_RUNTIME_DIR "/tmp"

RUN python3 -m pip install --upgrade pip

WORKDIR /
RUN mkdir -p /python_object_detection_server
COPY . /python_object_detection_server

RUN bash /python_object_detection_server/setting-scripts/install_dependencies.sh

RUN pip install jupyter pandas fastapi[all] python-multipart jupyter fastapi_utils loguru onvif2-zeep icecream pycuda

RUN bash /python_object_detection_server/setting-scripts/install_ffmpeg.sh

RUN bash /python_object_detection_server/setting-scripts/install_OpenCV.sh

