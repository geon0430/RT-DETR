# RT-DETR
[English](https://github.com/geon0430/RT-DETR/blob/main/README_en.md)
- Paddle 모델을 ONNX모델로 변환하여 Paddle package를 사용하지 않고 Pytorch나 TensorRT로 사용하는 것이 목적
- ONNX는 서로 다른 딥러닝 프레임워크에서 동일한 모델을 읽어와서 사용할 수 있도록 표준화 하는 패키지로 ONNX 모델을 사용할 경우 우리가 직접 소스코드를 작성하여  Paddle package를 사용하지 않고 모델을 사용 할 수 있음


### 1.clone 
```
git clone https://github.com/geon0430/RT-DETR.git
```

### 2. model pdparams 다운
- 원하는 모델 pdparams 다운 ([[https://github.com/lyuwenyu/RT-DETR](https://github.com/PaddlePaddle/PaddleDetection)](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.7/configs/rtdetr) 사이트에서 원하는 모델의 pdparams 다운

### 3. detect predict
```
python tools/infer.py
 -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
 -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams
 --infer_img=/./input_images/0001.jpg --output_dir ./outputs
```

### 4. onnx model build
- onxx 모델 빌드 하기 위해선 pdparams,  model.pdmodel 을 먼저 만들어야함
- pdparams,  model.pdmodel 생성 후 onnx 빌드
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
