#!/bin/bash

trtexec --onnx=./model/rtdetr_hgnetv2_l_6x_coco/rtdetr_r50vd_6x_coco.onnx\
        --shapes=image:3x3x640x640 \
        --saveEngine=./model/rtdetr_hgnetv2_l_6x_coco/rtdetr_r50vd_6x_coco.trt \
