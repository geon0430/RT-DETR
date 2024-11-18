# RT-DETR
[English](https://github.com/geon0430/RT-DETR/blob/main/README_en.md)
- Paddle 모델을 ONNX모델로 변환하여 Paddle package를 사용하지 않고 Pytorch나 TensorRT로 사용하는 것이 목적
- ONNX는 서로 다른 딥러닝 프레임워크에서 동일한 모델을 읽어와서 사용할 수 있도록 표준화 하는 패키지로 ONNX 모델을 사용할 경우 우리가 직접 소스코드를 작성하여  Paddle package를 사용하지 않고 모델을 사용 할 수 있음


### 1.clone 
```
git clone https://github.com/geon0430/RT-DETR.git
```


### 4. onnx model build
- onxx 모델 빌드 하기 위해선 pdparams,  model.pdmodel 을 먼저 만들어야함
- pdparams,  model.pdmodel 생성 후 onnx 빌드
```
bash create_pdmodel.sh
bash create_onnx.sh
```
### 5. tensorrt build
- onnx모델을 만든 후 trt 모델 빌드
```
bash create_tensorrt.sh
```
- trt 모델 입력 타입
```
input_data = {
                'image': pp_image
                'scale_factor'
                'im_shape'
            }
```
