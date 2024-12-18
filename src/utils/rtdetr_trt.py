## tensorrt 8.5.2
import tensorrt as trt
import cupy as cp
import numpy as np
import os
from typing import Dict, Tuple, Any, List, Union

trt.nptype = lambda dtype: {
    trt.bool: np.bool_,
    trt.int8: np.int8,
    trt.int32: np.int32,
    trt.float16: np.float16,
    trt.float32: np.float32
}[dtype]


class RT_DETR_CP:
    def __init__(self, gpu: int, model_path: str, input_shape: Tuple[int, ...], datatype: cp.dtype = cp.float32, logger: Any = None):
        self.gpu = gpu
        try:
            with cp.cuda.Device(self.gpu).use():
                self.TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
                self.engine = self.load_engine(model_path)
                self.context = self.engine.create_execution_context()
                self.inputs, self.outputs = self.allocate_buffers()
        except cp.cuda.runtime.CUDARuntimeError as e:
            max_gpu = cp.cuda.runtime.getDeviceCount()
            if logger:
                logger.error(f"RT_DETR_CP | Select gpu: {gpu} exceeds the maximum number of GPUs, Num GPUs: {max_gpu}")
            assert False, f"RT_DETR_CP | Select gpu: {gpu} exceeds the maximum number of GPUs, Num GPUs: {max_gpu}"

    def allocate_buffers(self) -> Tuple[Dict[str, Tuple[cp.ndarray, cp.ndarray]], Dict[str, Tuple[cp.ndarray, cp.ndarray]]]:
        inputs = {}
        outputs = {}
        with cp.cuda.Device(self.gpu).use():
            for binding in range(self.engine.num_bindings):
                tensor_name = self.engine.get_tensor_name(binding)
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                dims = self.engine.get_tensor_shape(tensor_name)
                shape = tuple(dims[i] for i in range(len(dims)))
                host_mem = cp.empty(shape, dtype)
                device_mem = cp.empty(shape, dtype)

                if self.engine.binding_is_input(binding):
                    inputs[tensor_name] = (host_mem, device_mem)
                else:
                    outputs[tensor_name] = (host_mem, device_mem)
        return inputs, outputs

    def predict(self, inputs: Dict[str, Union[cp.ndarray, cp.ndarray]]) -> List[cp.ndarray]:
        results = []
        with cp.cuda.Device(self.gpu).use():
            for input_name, data in inputs.items():
                if input_name in self.inputs:
                    host_mem, device_mem = self.inputs[input_name]
                    data_cp = cp.asarray(data)
                    host_mem[:] = data_cp
                    device_mem[:] = host_mem

            input_buffers = [buf[1].data.ptr for buf in self.inputs.values()]
            output_buffers = [buf[1].data.ptr for buf in self.outputs.values()]
            self.context.execute_v2(input_buffers + output_buffers)

            for binding in range(self.engine.num_bindings):
                if not self.engine.binding_is_input(binding):
                    tensor_name = self.engine.get_tensor_name(binding)
                    host_mem, device_mem = self.outputs[tensor_name]
                    host_mem[:] = device_mem
                    results.append(host_mem.copy())

        return results

    def load_engine(self, engine_file_path: str) -> trt.ICudaEngine:
        if not os.path.exists(engine_file_path):
            raise FileNotFoundError(f"Engine Error | Model path not found: {engine_file_path}")
        with open(engine_file_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise Exception("Failed to deserialize the TensorRT engine.")
            return engine


# ## tensorrt 10.3.0
# import tensorrt as trt
# import cupy as cp
# import numpy as np
# import os
# from typing import Dict, Tuple, Any, List, Union

# trt.nptype = lambda dtype: {
#     trt.bool: np.bool_,
#     trt.int8: np.int8,
#     trt.int32: np.int32,
#     trt.float16: np.float16,
#     trt.float32: np.float32
# }[dtype]


# class RT_DETR_CP:
#     def __init__(self, gpu: int, model_path: str, input_shape: Tuple[int, ...], datatype: cp.dtype = cp.float32, logger: Any = None):
#         self.gpu = gpu
#         try:
#             with cp.cuda.Device(self.gpu).use():
#                 self.TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
#                 self.engine = self.load_engine(model_path)
#                 self.context = self.engine.create_execution_context()
#                 self.inputs, self.outputs = self.allocate_buffers()
#         except cp.cuda.runtime.CUDARuntimeError as e:
#             max_gpu = cp.cuda.runtime.getDeviceCount()
#             if logger:
#                 logger.error(f"RT_DETR_CP | Select gpu: {gpu} exceeds the maximum number of GPUs, Num GPUs: {max_gpu}")
#             assert False, f"RT_DETR_CP | Select gpu: {gpu} exceeds the maximum number of GPUs, Num GPUs: {max_gpu}"

#     def allocate_buffers(self) -> Tuple[Dict[str, Tuple[cp.ndarray, cp.ndarray]], Dict[str, Tuple[cp.ndarray, cp.ndarray]]]:
#         inputs = {}
#         outputs = {}
#         with cp.cuda.Device(self.gpu).use():
#             num_io_tensors = self.engine.num_io_tensors
#             for i in range(num_io_tensors):
#                 tensor_name = self.engine.get_tensor_name(i)
#                 dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
#                 dims = self.engine.get_tensor_shape(tensor_name)
#                 shape = tuple(dims)
#                 host_mem = cp.empty(shape, dtype)
#                 device_mem = cp.empty(shape, dtype)

#                 if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
#                     inputs[tensor_name] = (host_mem, device_mem)
#                 else:
#                     outputs[tensor_name] = (host_mem, device_mem)
#         return inputs, outputs

#     def predict(self, inputs: Dict[str, Union[cp.ndarray, cp.ndarray]]) -> List[cp.ndarray]:
#         results = []
#         with cp.cuda.Device(self.gpu).use():
#             for input_name, data in inputs.items():
#                 if input_name in self.inputs:
#                     host_mem, device_mem = self.inputs[input_name]
#                     data_cp = cp.asarray(data)
#                     host_mem[:] = data_cp
#                     device_mem[:] = host_mem

#             for input_name, (host_mem, device_mem) in self.inputs.items():
#                 self.context.set_tensor_address(input_name, device_mem.data.ptr)
#             for output_name, (host_mem, device_mem) in self.outputs.items():
#                 self.context.set_tensor_address(output_name, device_mem.data.ptr)

#             # 실행: execute_async_v3 사용 (stream은 0 사용)
#             self.context.execute_async_v3(0)
#             # 동기화
#             cp.cuda.runtime.deviceSynchronize()

#             # 결과 복사
#             for output_name, (host_mem, device_mem) in self.outputs.items():
#                 host_mem[:] = device_mem
#                 results.append(host_mem.copy())

#         return results

#     def load_engine(self, engine_file_path: str) -> trt.ICudaEngine:
#         if not os.path.exists(engine_file_path):
#             raise FileNotFoundError(f"Engine Error | Model path not found: {engine_file_path}")
#         with open(engine_file_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
#             engine = runtime.deserialize_cuda_engine(f.read())
#             if engine is None:
#                 raise Exception("Failed to deserialize the TensorRT engine.")
#             return engine
