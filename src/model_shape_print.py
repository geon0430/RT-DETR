import onnxruntime as ort

def print_model_input_names_and_shapes(model_path):
    session = ort.InferenceSession(model_path)
    for input in session.get_inputs():
        print("Name:", input.name, "Shape:", input.shape)

model_path = "./model/rtdetr_hgnetv2_l_6x_coco/rtdetr_r50vd_6x_coco.onnx"


print_model_input_names_and_shapes(model_path)
