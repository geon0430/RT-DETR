# RT-DETR
[English](https://github.com/geon0430/RT-DETR/blob/main/README_en.md)
- 라이센스 문제 없는 PaddleDetection의 RT-DETR 모델로 custom_dataset으로 train하여 onnx, tensorrt로 사용


## Train 
### 1. 학습을 진행하기 위해서 먼저 fine turning 모델을 선택한다.
| Model | Epoch | Backbone  | Input shape | $AP^{val}$ | $AP^{val}_{50}$| Params(M) | FLOPs(G) |  T4 TensorRT FP16(FPS) | Pretrained Model | config |
|:--------------:|:-----:|:----------:| :-------:|:--------------------------:|:---------------------------:|:---------:|:--------:| :---------------------: |:------------------------------------------------------------------------------------:|:-------------------------------------------:|
| RT-DETR-R18 | 6x |  ResNet-18 | 640 | 46.5 | 63.8 | 20 | 60 | 217 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r18vd_dec3_6x_coco.pdparams) | [config](./rtdetr_r18vd_6x_coco.yml)
| RT-DETR-R34 | 6x |  ResNet-34 | 640 | 48.9 | 66.8 | 31 | 92 | 161 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r34vd_dec4_6x_coco.pdparams) | [config](./rtdetr_r34vd_6x_coco.yml)
| RT-DETR-R50-m | 6x |  ResNet-50 | 640 | 51.3 | 69.6 | 36 | 100 | 145 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_m_6x_coco.pdparams) | [config](./rtdetr_r50vd_m_6x_coco.yml)
| RT-DETR-R50 | 6x |  ResNet-50 | 640 | 53.1 | 71.3 | 42 | 136 | 108 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams) | [config](./rtdetr_r50vd_6x_coco.yml)
| RT-DETR-R101 | 6x |  ResNet-101 | 640 | 54.3 | 72.7 | 76 | 259 | 74 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r101vd_6x_coco.pdparams) | [config](./rtdetr_r101vd_6x_coco.yml)
| RT-DETR-L | 6x |  HGNetv2 | 640 | 53.0 | 71.6 | 32 | 110 | 114 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_hgnetv2_l_6x_coco.pdparams) | [config](rtdetr_hgnetv2_l_6x_coco.yml)
| RT-DETR-X | 6x |  HGNetv2 | 640 | 54.8 | 73.1 | 67 | 234 | 74 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_hgnetv2_x_6x_coco.pdparams) | [config](rtdetr_hgnetv2_x_6x_coco.yml)
| RT-DETR-H | 6x |  HGNetv2 | 640 | 56.3 | 74.8 | 123 | 490 | 40 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_hgnetv2_h_6x_coco.pdparams) | [config](rtdetr_hgnetv2_h_6x_coco.yml)


###  2.  fine turning 모델을 선택했으면 "configs/rtdetr/rtdetr_r50vd_6x_coco.yml" 경로에 들어가 모델의 yaml파일을 찾아들어간다
- coco datasets으로 fine turning된 모델을 만들려면 _BASE_: [ '../datasets/coco_detection.yml' ] 으로 입력한다.
- custom datasets으로 특정 객체인식 모델을 만들려면  _BASE_: [ '../datasets/custom_dataset.yml' ] 으로 입력한다.

##  custom model 학습 방법 
### 2, dataset 구성
- coco dataset 형식으로 데이터셋을 준비해야함
```
 dataset/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   ├── instances_test.json
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
├── val/
│   ├── image3.jpg
│   ├── image4.jpg
├── test/
    ├── image5.jpg
    ├── image6.jpg
```
### 3. dataset yaml 작성
- src/PaddleDetection/configs/datasets/custom_dataset.yml 파일에서 데이터셋 위치를 지정함
```
 ### src/PaddleDetection/configs/datasets/custom_dataset.yml
metric: COCO
num_classes: 4

TrainDataset:
    name : COCODataSet
    image_dir: train
    anno_path: annotations/train_datasets.json
    dataset_dir: /rt-detr-paddle-build-onnx-tensorrt/src/PaddleDetection/dataset
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
    name : COCODataSet
    image_dir: val
    anno_path: annotations/valid_datasets.json
    dataset_dir: /rt-detr-paddle-build-onnx-tensorrt/src/PaddleDetection/dataset
    allow_empty: true

TestDataset:
    name : COCODataSet
    image_dir: test
    anno_path: annotations/test_datasets.json
    dataset_dir: /rt-detr-paddle-build-onnx-tensorrt/src/PaddleDetection/dataset 
```

### 4. train
- wandb를 사용할려면 학습을 진행하기 전 wandb api를 등록 후 --use_wandb True를 추가한다.
```
wandb login
```

- 학습 결과로 pdparams 가 생성됨 
```
cd /rt-detr-paddle-build-onnx-tensorrt/src/PaddleDetection
export CUDA_VISIBLE_DEVICES=0,1 ## 사용할 gpu 압력
python -m paddle.distributed.launch --gpus 0,1 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml --fleet --eval
```

## Build
### 1. onnx model build
- onxx 모델 빌드 하기 위해선 pdparams,  model.pdmodel 을 먼저 만들어야함
- pdparams,  model.pdmodel 생성 후 onnx 빌드
```
bash create_pdmodel.sh
bash create_onnx.sh
```
### 2. tensorrt build
- onnx모델을 만든 후 trt 모델 빌드
```
bash create_tensorrt.sh
```
### 3. trt model predict test
```
python utils/trt_predict.py
```
|Input_data |Result_data |
|:--------------:|:--------------:|
| ![1](https://github.com/user-attachments/assets/eaf1e9a5-379c-46b5-bdc7-d52845cd6667) | ![result](https://github.com/user-attachments/assets/d190f415-9447-49bb-bc6d-ab3141648870) |
