import onnxruntime as ort
import numpy as np
import time
import os
import argparse

def benchmark(model_path, provider, iterations=50, warmups=5):
    print(f"\nBenchmarking {model_path} with {provider}...")
    
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 0
    sess_options.log_verbosity_level = 1
    
    # Provider options for CoreML
    provider_options = {}
    if provider == 'CoreMLExecutionProvider':
        provider_options = {
            'MLComputeUnits': 'ANE_ONLY', 
        }
    
    try:
        session = ort.InferenceSession(
            model_path, 
            sess_options, 
            providers=[(provider, provider_options)] if provider_options else [provider]
        )
    except Exception as e:
        print(f"Failed to create session with {provider}: {e}")
        return

    inputs = session.get_inputs()
    input_feed = {}
    for inp in inputs:
        input_name = inp.name
        input_shape = inp.shape
        input_type = inp.type
        
        # Map ONNX types to numpy types
        type_map = {
            'tensor(float16)': np.float16,
            'tensor(float)': np.float32,
            'tensor(int64)': np.int64,
        }
        np_type = type_map.get(input_type, np.float32)
        
        # Handle dynamic axes
        processed_shape = []
        for s in input_shape:
            if isinstance(s, str):
                processed_shape.append(1)
            else:
                processed_shape.append(s)
        
        input_feed[input_name] = np.random.randn(*processed_shape).astype(np_type)
        if np_type == np.int64:
            input_feed[input_name] = np.ones(processed_shape).astype(np_type)
    
    # Warmup
    print(f"Warming up for {warmups} iterations...")
    for _ in range(warmups):
        session.run(None, input_feed)
        
    # Benchmark
    print(f"Running benchmark for {iterations} iterations...")
    start_time = time.perf_counter()
    for _ in range(iterations):
        session.run(None, input_feed)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = (total_time / iterations) * 1000
    fps = iterations / total_time
    
    print(f"Average Latency: {avg_time:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    return avg_time, fps

def benchmark_all(model_path, iterations=50):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    providers = [
        ('CoreMLExecutionProvider', {'MLComputeUnits': 'ANE_ONLY'}),
    ]
    
    results = {}
    for provider_name, options in providers:
        print(f"\n--- Provider: {provider_name} ---")
        try:
            sess_options = ort.SessionOptions()
            session = ort.InferenceSession(model_path, sess_options, providers=[(provider_name, options)] if options else [provider_name])
            
            # Check ANE support
            if provider_name == 'CoreMLExecutionProvider':
                # We can't easily get the support count here without log capturing, but we'll see it in logs if enabled
                pass
                
            res = benchmark(model_path, provider_name, iterations=iterations, session=session)
            results[provider_name] = res
        except Exception as e:
            print(f"Failed to benchmark {provider_name}: {e}")
            
    return results

def benchmark(model_path, provider, iterations=50, warmups=5, session=None):
    if session is None:
        sess_options = ort.SessionOptions()
        session = ort.InferenceSession(model_path, sess_options, providers=[provider])

    inputs = session.get_inputs()
    input_feed = {}
    for inp in inputs:
        input_name = inp.name
        input_shape = inp.shape
        input_type = inp.type
        
        type_map = {
            'tensor(float16)': np.float16,
            'tensor(float)': np.float32,
            'tensor(int64)': np.int64,
        }
        np_type = type_map.get(input_type, np.float32)
        
        processed_shape = []
        for s in input_shape:
            if isinstance(s, str) or s is None or s < 0:
                processed_shape.append(1) # Default batch size 1
            else:
                processed_shape.append(s)
        
        input_feed[input_name] = np.random.randn(*processed_shape).astype(np_type)
        if np_type == np.int64:
            input_feed[input_name] = np.ones(processed_shape).astype(np_type)
    
    # Warmup
    for _ in range(warmups):
        session.run(None, input_feed)
        
    start_time = time.perf_counter()
    for _ in range(iterations):
        session.run(None, input_feed)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = (total_time / iterations) * 1000
    fps = iterations / total_time
    
    print(f"  Latency: {avg_time:.2f} ms | Throughput: {fps:.2f} FPS")
    return avg_time, fps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/table-transformer-structure-recognition-ane.onnx")
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    
    benchmark_all(args.model, iterations=args.iterations)
