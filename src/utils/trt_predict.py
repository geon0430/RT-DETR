import cv2
import yaml
import random
import numpy as np
import cupy as cp
import configparser
from typing import Any, Tuple, Dict, List
from rtdetr_trt import RT_DETR_CP
import cucim.skimage.transform
from process import preprocess, postprocess
from logger import setup_logger

def rtdetr_trt_predict():
    
    input_image_path = "/rt-detr-paddle-build-onnx-tensorrt/src/volume/1.jpg"
    output_image_path = "result.jpg"
    
    config_path = "/rt-detr-paddle-build-onnx-tensorrt/src/utils/config.ini"
    config = configparser.ConfigParser()
    config.read(config_path)
    logger = setup_logger(config)
    
    ## RTDETR ##
    batch: int = config.getint('RTDETR','BATCH')
    channel: int = config.getint('RTDETR','CHANNEL')
    gpu: int = config.getint('RTDETR','GPU')
    rtdetr_model_path: str = config['RTDETR']['MODEL_PATH']
    config_path: str = config['RTDETR']['CLASS_CONFIG_PATH']
    model_height: int = config.getint('RTDETR','MODEL_HEIGHT')
    model_width: int = config.getint('RTDETR','MODEL_WIDTH')
    
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    label_list = config['label_list']

    with cp.cuda.Device(gpu):
        scale_factors: cp.ndarray = cp.array([1.0, 1.0], dtype=cp.float32).reshape(1, 2)
        im_shape: cp.ndarray = cp.array([model_width, model_height], dtype=cp.float32).reshape(1, 2)

        rtdetr = RT_DETR_CP(
            gpu=gpu,
            model_path=rtdetr_model_path,
            input_shape=(batch, channel, model_height, model_width),
            datatype=cp.float32,
        )

        Input_data = cv2.imread(input_image_path)
        logger.info(f"Input_data : {Input_data.size}, shape: {Input_data.shape}, type: {type(Input_data)}{Input_data.dtype}")
        
        Input_height, Input_width, _ = Input_data.shape
        scale_width: float = Input_width / model_width
        scale_height: float = Input_height / model_height
        logger.info(f"Input_width: {Input_width}, Input_height : {Input_height}, scale_width : {scale_width}, scale_height: {scale_height}")

        Input_data_cp = cp.asarray(Input_data)

        resized_BGR_image_gpu = cucim.skimage.transform.resize(
            Input_data_cp,
            (model_height, model_width, channel),
            preserve_range=True,
            order=1
        )
        logger.info(f"Resize Data : {resized_BGR_image_gpu.size}, shape: {resized_BGR_image_gpu.shape}, type: {type(resized_BGR_image_gpu)}{resized_BGR_image_gpu.dtype}")
        # Preprocess
        pp_image = preprocess(
            image_data=resized_BGR_image_gpu,
            dtype=cp.float32,
            normalize_factor=255.0,
            channel_first=True,
            batched=True,
        )
        logger.info(f"Preprocess Data : {pp_image.size}, shape: {pp_image.shape}, type: {type(pp_image)}{pp_image.dtype}")

        input_data: Dict[str, cp.ndarray] = {
            'image': pp_image,
            'scale_factor': scale_factors,
            'im_shape': im_shape
        }

        # Prediction
        outputs = rtdetr.predict(input_data)

        # Postprocess
        logger.info(f"Predict Data : {outputs[1].size}, shape: {outputs[1].shape}, type: {type(outputs[1])}{outputs[1].dtype}")
        postprocess_data: list = postprocess(
            output_data=outputs[1],
            scale_width=scale_width,
            scale_height=scale_height,
            score_threshold=0.5,
            iou_threshold=0.5,
        )
        for bbox in postprocess_data:
            class_id, score, x_min, y_min, x_max, y_max = bbox

            class_label = label_list[int(class_id)]
            label = f"{class_label} {score:.2f}"
            color = [random.randint(0, 255) for _ in range(3)]  

            cv2.rectangle(Input_data, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(Input_data, (int(x_min), int(y_min) - text_height - baseline), (int(x_min) + text_width, int(y_min)), color, cv2.FILLED)
            cv2.putText(Input_data, label, (int(x_min), int(y_min) - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imwrite(output_image_path, Input_data)
        print(f"Result saved to {output_image_path}")

if __name__ == "__main__":
    
    rtdetr_trt_predict()
