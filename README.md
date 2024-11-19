# RT-DETR
[English](https://github.com/geon0430/rt-detr-paddle-build-onnx-tensorrt/blob/main/README_en.md)
- Paddle 모델을 ONNX모델로 변환하여 Paddle package를 사용하지 않고 Onnx 또는 TensorRT로 사용하는 것이 목적
- ONNX는 서로 다른 딥러닝 프레임워크에서 동일한 모델을 읽어와서 사용할 수 있도록 표준화 하는 패키지로 ONNX 모델을 사용할 경우 우리가 직접 소스코드를 작성하여  Paddle package를 사용하지 않고 모델을 사용 할 수 있음


### 1.clone 
```
git clone https://github.com/geon0430/rt-detr-paddle-build-onnx-tensorrt.git
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
### 6. trt model predict test
```
python utils/trt_predict.py
```
|Input_data |Result_data |
|:--------------:|:--------------:|
| ![1](https://github.com/user-attachments/assets/eaf1e9a5-379c-46b5-bdc7-d52845cd6667) | ![result](https://github.com/user-attachments/assets/d190f415-9447-49bb-bc6d-ab3141648870) |
