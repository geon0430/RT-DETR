# RT-DETR
------
- The purpose is to convert the Paddle model to ONNX model and use it with Pytorch or TensorRT without using the Paddle package.
- ONNX is a package that standardizes so that the same model can be read and used in different deep learning frameworks. When using the ONNX model, we can write the source code ourselves and use the model without using the Paddle package.


### 1.clone 
```
git clone https://github.com/geon0430/rt-detr-paddle-build-onnx-tensorrt.git
```

### 4. onnx model build
- To build an onxx model, you must first create pdparams and model.pdmodel.
- Build onnx after creating pdparams and model.pdmodel
```
bash create_pdmodel.sh
bash create_onnx.sh
```
### 5. tensorrt build
- To build an tensorrt model, you must first create onnx
```
bash create_tensorrt.sh
```
### 6. trt model predict test
```
python utils/trt_predict.py
```
