from time import perf_counter
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt


def create_session(model_path, use_cuda=True):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # providers = ["CUDAExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    providers = [
        ("CoreMLExecutionProvider", {"use_ane": True}),
        "CPUExecutionProvider",
    ]

    session = ort.InferenceSession(
        model_path, sess_options=session_options, providers=providers
    )
    return session


def run_inference(session, input_tensor):
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_tensor})


def measure_inference_batch(session, batch_size: int, repeats: int):
    input_tensor = np.random.rand(batch_size, 3, 1024, 1024).astype(np.float32)

    # Warm-up
    run_inference(session, input_tensor)

    latencies = []
    # Timed loop
    start_time = perf_counter()
    for _ in range(repeats):
        iteration_start = perf_counter()
        _ = run_inference(session, input_tensor)
        iteration_end = perf_counter()
        latencies.append(iteration_end - iteration_start)
    end_time = perf_counter()

    total_duration = end_time - start_time
    avg_latency = np.mean(latencies)
    throughput = (repeats * batch_size) / total_duration
    return total_duration, throughput, avg_latency


def measure_inference_single(session, batch_size, repeats):
    input_tensor = np.random.rand(1, 3, 1024, 1024).astype(np.float32)
    run_inference(session, input_tensor)  # warm-up

    total_inferences = batch_size * repeats
    latencies = []

    start_time = perf_counter()
    for _ in range(total_inferences):
        iteration_start = perf_counter()
        _ = run_inference(session, input_tensor)
        iteration_end = perf_counter()
        latencies.append(iteration_end - iteration_start)
    end_time = perf_counter()

    total_duration = end_time - start_time
    avg_latency = np.mean(latencies)
    throughput = total_inferences / total_duration
    return total_duration, throughput, avg_latency


if __name__ == "__main__":
    # Set up models and sessions
    model_name = "yolov8s-doclaynet"
    single_model_path = f"../models/{model_name}.onnx"
    single_session = create_session(single_model_path, use_cuda=True)

    batch_sizes = [2, 4, 8, 32]
    N_values = np.logspace(0, np.log10(20), num=10)
    N_values = np.unique(N_values.astype(int))

    # Storage for results
    results = {
        batch_size: {
            "batch_throughputs": [],
            "batch_latencies": [],
            "single_throughputs": [],
            "single_latencies": [],
        }
        for batch_size in batch_sizes
    }

    for batch_size in batch_sizes:
        # Construct the batch model path
        batch_model_path = f"../models/{model_name}_batchsize{batch_size}.onnx"
        batch_session = create_session(batch_model_path, use_cuda=True)

        for N in N_values:
            _, throughput, latency = measure_inference_batch(
                batch_session, batch_size, repeats=N
            )
            results[batch_size]["batch_throughputs"].append(throughput)
            results[batch_size]["batch_latencies"].append(latency)

            _, throughput_s, latency_s = measure_inference_single(
                single_session, batch_size, repeats=N
            )
            results[batch_size]["single_throughputs"].append(throughput_s)
            results[batch_size]["single_latencies"].append(latency_s)

        # Clean up after each batch size
        del batch_session

    # Plot throughput results
    plt.figure(figsize=(12, 8))
    for batch_size in batch_sizes:
        plt.plot(
            N_values,
            results[batch_size]["batch_throughputs"],
            label=f"Batched Model (batch={batch_size})",
        )
        plt.plot(
            N_values,
            results[batch_size]["single_throughputs"],
            "--",
            label=f"Single-Inference (batch={batch_size})",
        )

    plt.xlabel("N (number of inference repeats)")
    plt.ylabel("Throughput (pages/sec)")
    plt.title("ONNX Inference Throughput Comparison Across Batch Sizes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("throughput_comparison.png")
    plt.show()

    # Plot latency results
    plt.figure(figsize=(12, 8))
    for batch_size in batch_sizes:
        plt.plot(
            N_values,
            results[batch_size]["batch_latencies"],
            label=f"Batched Model (batch={batch_size})",
        )
        plt.plot(
            N_values,
            results[batch_size]["single_latencies"],
            "--",
            label=f"Single-Inference (batch={batch_size})",
        )

    plt.xlabel("N (number of inference repeats)")
    plt.ylabel("Average Latency (seconds)")
    plt.title("ONNX Inference Latency Comparison Across Batch Sizes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("latency_comparison.png")
    plt.show()

    # Clean up
    del single_session
