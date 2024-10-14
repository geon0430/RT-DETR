#!/bin/bash

python ./PaddleDetection/tools/infer.py \
 -c PaddleDetection/configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
 -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams \
 --infer_img ./volume/1.jpg --output_dir ./volume
