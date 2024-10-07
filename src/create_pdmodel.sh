#!/bin/bash

python rtdetr_paddle/tools/export_model.py \
  -c ./rtdetr_paddle/configs/rtdetr/rtdetr_hgnetv2_l_6x_coco.yml \
  -o ./model/rtdetr_hgnetv2_l_6x_coco.pdparams \
  --output_dir=./model/
