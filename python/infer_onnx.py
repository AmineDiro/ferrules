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
    providers = [
        ("CoreMLExecutionProvider", {"use_ane": True}),
        "CPUExecutionProvider",
    ]

    # providers = ["CUDAExecutionProvider"]

    session = ort.InferenceSession(
        model_path, sess_options=session_options, providers=providers
    )
    return session


def run_inference(session, input_tensor):
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_tensor})


if __name__ == "__main__":
    BATCH_SIZE = 32
    N = 10
    ##### BATCH
    onnx_model_path = f"../models/yolov8s-doclaynet_batchsize{BATCH_SIZE}.onnx"
    batch_session = create_session(onnx_model_path)

    input_tensor = np.random.rand(BATCH_SIZE, 3, 1024, 1024).astype(np.float32)

    _ = run_inference(batch_session, input_tensor)
    s = perf_counter()
    for i in range(N):
        outputs = run_inference(batch_session, input_tensor)
    e = perf_counter()
    duration = e - s
    print(
        f"Model {onnx_model_path} took: {duration:.2f}s. {N*BATCH_SIZE/duration:.2f} page/s"
    )
    del batch_session

    ##### SINGLE
    onnx_model_path = "../models/yolov8s-doclaynet.onnx"
    single_batch_session = create_session(onnx_model_path)
    input_tensor = np.random.rand(1, 3, 1024, 1024).astype(np.float32)
    _ = run_inference(single_batch_session, input_tensor)

    s = perf_counter()
    for _ in range(BATCH_SIZE * N):
        _ = run_inference(single_batch_session, input_tensor)
    e = perf_counter()
    duration = e - s
    print(
        f"Single threaded model {onnx_model_path} took: {duration:.2f}s. {N*BATCH_SIZE/duration:.2f} page/s"
    )

    ## MULTITHREADED
    s = perf_counter()
    with concurrent.futures.ThreadPoolExecutor(10) as executor:
        futures = [
            executor.submit(run_inference, single_batch_session, input_tensor)
            for _ in range(BATCH_SIZE * N)
        ]
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]

    e = perf_counter()
    duration = e - s
    print(
        f"Multithreaded single_batch model {onnx_model_path} took: {duration:.2f}s. {N*BATCH_SIZE/duration:.2f} page/s"
    )
