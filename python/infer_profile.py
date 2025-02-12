
import numpy as np
import onnxruntime as ort

onnx_model_path = "../models/yolov8s-doclaynet.onnx"

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session_options.enable_profiling = True

providers = ["CUDAExecutionProvider"]

session = ort.InferenceSession(
    onnx_model_path, sess_options=session_options, providers=providers
)
input_name = session.get_inputs()[0].name
input_tensor = np.random.rand(1, 3, 1024, 1024).astype(np.float32)
output = session.run(None, {input_name: input_tensor})

