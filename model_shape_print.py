import onnxruntime as ort

def print_model_input_names_and_shapes(model_path):
    session = ort.InferenceSession(model_path)
    for input in session.get_inputs():
        print("Name:", input.name, "Shape:", input.shape)

model_path = "/python_object_detection_server/padder/test/model/test.onnx"


print_model_input_names_and_shapes(model_path)
