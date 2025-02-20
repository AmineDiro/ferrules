import concurrent.futures
from time import perf_counter

import numpy as np
import onnxruntime as ort


def create_session(model_path):
    # Create session options
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # session_options.enable_profiling = True

    # Use CoreMLExecutionProvider with settings for ANE only
    # providers = ["CPUExecutionProvider"]
    # providers = [
    #     ("CoreMLExecutionProvider", {"use_ane": True}),
    #     "CPUExecutionProvider",
    # ]

    providers = ["CUDAExecutionProvider"]

    session = ort.InferenceSession(
        model_path, sess_options=session_options, providers=providers
    )
    return session


def run_inference(session, input_tensor):
    # Get model input name
    input_name = session.get_inputs()[0].name
    # Run inference
    output = session.run(None, {input_name: input_tensor})
    return output


if __name__ == "__main__":
    ##### BATCH
    onnx_model_path = "../models/yolov8s-doclaynet-batch-16.onnx"
    batch_session = create_session(onnx_model_path)

    input_tensor = np.random.rand(16, 3, 1024, 1024).astype(np.float32)

    # _ = run_inference(batch_session, input_tensor)
    s = perf_counter()
    for i in range(1):
        outputs = run_inference(batch_session, input_tensor)
    e = perf_counter()
    print(f"Model {onnx_model_path} took: {e-s:.2f}s")

    ##### SINGLE
    # onnx_model_path = "../models/yolov8s-doclaynet.onnx"
    # single_batch_session = create_session(onnx_model_path)
    # input_tensor = np.random.rand(1, 3, 1024, 1024).astype(np.float32)
    # # _ = run_inference(single_batch_session, input_tensor)
    # batch_size = 16  # Total number of inferences to run
    # s = perf_counter()
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [
    #         executor.submit(run_inference, single_batch_session, input_tensor)
    #         for _ in range(batch_size)
    #     ]
    #     results = [
    #         future.result() for future in concurrent.futures.as_completed(futures)
    #     ]

    # e = perf_counter()
    # print(f"Model {onnx_model_path} took: {e-s:.2f}s")
