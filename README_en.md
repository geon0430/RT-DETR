
# RT-DETR
- Train PaddleDetection's RT-DETR model with a custom dataset for onnx and TensorRT deployment without licensing issues.

---

## Training

### 1. Select a Pretrained Model for Fine-tuning
| Model          | Epoch | Backbone    | Input Shape | $AP^{val}$ | $AP^{val}_{50}$ | Params (M) | FLOPs (G) | T4 TensorRT FP16 (FPS) | Pretrained Model | Config                                                                 |
|:--------------:|:-----:|:-----------:|:-----------:|:----------:|:---------------:|:----------:|:---------:|:----------------------:|:---------------------------------------------------------------------------:|:-------------------------------------------:|
| RT-DETR-R18    | 6x    | ResNet-18   | 640         | 46.5       | 63.8            | 20         | 60        | 217                    | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r18vd_dec3_6x_coco.pdparams) | [config](./rtdetr_r18vd_6x_coco.yml) |
| RT-DETR-R34    | 6x    | ResNet-34   | 640         | 48.9       | 66.8            | 31         | 92        | 161                    | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r34vd_dec4_6x_coco.pdparams) | [config](./rtdetr_r34vd_6x_coco.yml) |
| RT-DETR-R50-m  | 6x    | ResNet-50   | 640         | 51.3       | 69.6            | 36         | 100       | 145                    | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_m_6x_coco.pdparams) | [config](./rtdetr_r50vd_m_6x_coco.yml) |
| RT-DETR-R50    | 6x    | ResNet-50   | 640         | 53.1       | 71.3            | 42         | 136       | 108                    | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams) | [config](./rtdetr_r50vd_6x_coco.yml) |
| RT-DETR-R101   | 6x    | ResNet-101  | 640         | 54.3       | 72.7            | 76         | 259       | 74                     | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r101vd_6x_coco.pdparams) | [config](./rtdetr_r101vd_6x_coco.yml) |
| RT-DETR-L      | 6x    | HGNetv2     | 640         | 53.0       | 71.6            | 32         | 110       | 114                    | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_hgnetv2_l_6x_coco.pdparams) | [config](./rtdetr_hgnetv2_l_6x_coco.yml) |
| RT-DETR-X      | 6x    | HGNetv2     | 640         | 54.8       | 73.1            | 67         | 234       | 74                     | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_hgnetv2_x_6x_coco.pdparams) | [config](./rtdetr_hgnetv2_x_6x_coco.yml) |
| RT-DETR-H      | 6x    | HGNetv2     | 640         | 56.3       | 74.8            | 123        | 490       | 40                     | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_hgnetv2_h_6x_coco.pdparams) | [config](./rtdetr_hgnetv2_h_6x_coco.yml) |

---

### 2. Update Model Configuration
Locate the corresponding YAML configuration file, e.g., `"configs/rtdetr/rtdetr_r50vd_6x_coco.yml"`. 

- For **COCO datasets**, set:
  ```yaml
  _BASE_: [ '../datasets/coco_detection.yml' ]
  ```
- For **custom datasets**, set:
  ```yaml
  _BASE_: [ '../datasets/custom_dataset.yml' ]
  ```

---

## Custom Model Training

### 1. Dataset Preparation
Organize your dataset in the COCO format:

```plaintext
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

---

### 2. Dataset YAML Configuration
Specify the dataset's location in the YAML file, e.g., `src/PaddleDetection/configs/datasets/custom_dataset.yml`:

```yaml
metric: COCO
num_classes: 4

TrainDataset:
    name: COCODataSet
    image_dir: train
    anno_path: annotations/train_datasets.json
    dataset_dir: /rt-detr-paddle-build-onnx-tensorrt/src/PaddleDetection/dataset
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
    name: COCODataSet
    image_dir: val
    anno_path: annotations/valid_datasets.json
    dataset_dir: /rt-detr-paddle-build-onnx-tensorrt/src/PaddleDetection/dataset
    allow_empty: true

TestDataset:
    name: COCODataSet
    image_dir: test
    anno_path: annotations/test_datasets.json
    dataset_dir: /rt-detr-paddle-build-onnx-tensorrt/src/PaddleDetection/dataset
```

---

### 3. Start Training
If you wish to log training using **wandb**, log in with your API key and include `--use_wandb True`:

```bash
wandb login
```

Train the model:
```bash
cd /rt-detr-paddle-build-onnx-tensorrt/src/PaddleDetection
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch --gpus 0,1 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml --fleet --eval
```

The training process will output `.pdparams` files for the model.

---

## Building ONNX and TensorRT Models

### 1. ONNX Model Conversion
First, generate `.pdmodel` and `.pdparams` files. Then, convert to ONNX format:

```bash
bash create_pdmodel.sh
bash create_onnx.sh
```

---

### 2. TensorRT Model Conversion
After creating the ONNX model, build the TensorRT model:

```bash
bash create_tensorrt.sh
```

---

### 3. Testing the TensorRT Model
Run a prediction test on the TensorRT model:

```bash
python utils/trt_predict.py
```

| Input Data     | Result Data     |
|:--------------:|:---------------:|
| ![Input](https://github.com/user-attachments/assets/eaf1e9a5-379c-46b5-bdc7-d52845cd6667) | ![Result](https://github.com/user-attachments/assets/d190f415-9447-49bb-bc6d-ab3141648870) |
