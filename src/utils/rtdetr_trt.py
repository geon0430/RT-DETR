import cupy as cp
import tensorrt as trt
import os
from typing import Dict, Tuple, Any, List, Union

class RT_DETR_CP:
    def __init__(self, gpu: int, model_path: str, input_shape: Tuple[int, ...], datatype: cp.dtype = cp.float32, logger: Any = None):
        self.gpu = gpu
        with cp.cuda.Device(self.gpu):
            self.TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
            self.engine = self.load_engine(model_path)
            self.context = self.engine.create_execution_context()
            self.stream = cp.cuda.Stream() 
            try:
                self.inputs, self.outputs = self.allocate_buffers()
            except cp.cuda.runtime.CUDARuntimeError as e:
                max_gpu = cp.cuda.runtime.getDeviceCount()
                if logger:
                    logger.error(f"Select gpu: {gpu} exceeds the maximum number of GPUs, Num GPUs: {max_gpu}")
                assert False, f"Select gpu: {gpu} exceeds the maximum number of GPUs, Num GPUs: {max_gpu}"

    def allocate_buffers(self) -> Tuple[Dict[str, Tuple[cp.ndarray, cp.ndarray]], Dict[str, Tuple[cp.ndarray, cp.ndarray]]]:
        inputs = {}
        outputs = {}
        for binding in range(self.engine.num_bindings):
            tensor_name = self.engine.get_tensor_name(binding)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            dims = self.engine.get_tensor_shape(tensor_name)
            shape = tuple(dims[i] for i in range(len(dims)))
            host_mem = cp.empty(shape, dtype=dtype)
            device_mem = cp.empty(shape, dtype=dtype)

            if self.engine.binding_is_input(binding):
                inputs[tensor_name] = (host_mem, device_mem)
            else:
                outputs[tensor_name] = (host_mem, device_mem)
        return inputs, outputs

    def predict(self, inputs: Dict[str, Union[cp.ndarray, cp.ndarray]]) -> List[cp.ndarray]:
        results = []
        with cp.cuda.Device(self.gpu):
            with self.stream:
                for input_name, data in inputs.items():
                    if input_name in self.inputs:
                        host_mem, device_mem = self.inputs[input_name]
                        data_cp = cp.asarray(data)
                        cp.copyto(host_mem, data_cp)
                        cp.copyto(device_mem, host_mem)

                input_buffers = [buf[1].data.ptr for buf in self.inputs.values()]
                output_buffers = [buf[1].data.ptr for buf in self.outputs.values()]
                self.context.execute_async_v2(bindings=input_buffers + output_buffers, stream_handle=self.stream.ptr)

                for binding in range(self.engine.num_bindings):
                    if not self.engine.binding_is_input(binding):
                        tensor_name = self.engine.get_tensor_name(binding)
                        host_mem, device_mem = self.outputs[tensor_name]
                        cp.copyto(host_mem, device_mem)
                        results.append(device_mem) 

        return results


    def load_engine(self, engine_file_path: str) -> trt.ICudaEngine:
        if not os.path.exists(engine_file_path):
            raise FileNotFoundError(f"Engine Error | Model path not found: {engine_file_path}")
        with open(engine_file_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise Exception("Failed to deserialize the TensorRT engine.")
            return engine

    def __del__(self):
        if hasattr(self, 'stream') and self.stream:
            self.stream.synchronize() 
