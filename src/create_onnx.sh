#!/bin/bash

paddle2onnx --model_dir=./model/rtdetr_hgnetv2_l_6x_coco/ \
            --model_filename model.pdmodel  \
            --params_filename model.pdiparams \
            --opset_version 16 \
            --save_file ./model/rtdetr_hgnetv2_l_6x_coco/rtdetr_r50vd_6x_coco.onnx
