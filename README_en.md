# RT-DETR
------
- The purpose is to convert the Paddle model to ONNX model and use it with Pytorch or TensorRT without using the Paddle package.
- ONNX is a package that standardizes so that the same model can be read and used in different deep learning frameworks. When using the ONNX model, we can write the source code ourselves and use the model without using the Paddle package.


### 1.clone 
```
git clone https://github.com/geon0430/RT-DETR.git
```

### 2. model pdparams Download
- Desired model pdparams Download ([[https://github.com/lyuwenyu/RT-DETR](https://github.com/PaddlePaddle/PaddleDetection)](https://github.com/PaddlePaddle/PaddleDetection/tree /release/2.7/configs/rtdetr) Download the pdparams of the desired model from the site.

### 3. detect predict
```
python tools/infer.py
 -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
 -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams
 --infer_img=/./input_images/0001.jpg --output_dir ./outputs
```

### 4. onnx model build
- To build an onxx model, you must first create pdparams and model.pdmodel.
- Build onnx after creating pdparams and model.pdmodel
```
python tools/export_model.py \
  -c ./rtdetr_paddle/configs/rtdetr/rtdetr_hgnetv2_l_6x_coco.yml
  -o ./model/rtdetr_hgnetv2_l_6x_coco.pdparams \
  --output_dir=./model/

  paddle2onnx
    --model_dir=./model \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams \
    --opset_version 16 \
    --save_file ./model/rtdetr_r50vd_6x_coco.onnx
```
