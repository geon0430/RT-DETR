#!/bin/bash

python PaddleDetection/tools/export_model.py \
  -c ./PaddleDetection/configs/rtdetr/rtdetr_hgnetv2_l_6x_coco.yml \
  -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams trt=True \
  --output_dir=./model/
