import subprocess
import os
import argparse
import sys

def run_command(cmd):
    # Use the local .venv python
    python_path = "python/.venv/bin/python"
    if cmd.startswith("python "):
        cmd = cmd.replace("python ", f"{python_path} ", 1)
    print(f"Executing: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return stdout, stderr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    batch_sizes = [1, 2, 4, 8, 16]
    results = []

    for b in batch_sizes:
        print(f"\n--- Testing Batch Size: {b} ---")
        
        # 1. Export model
        export_cmd = f"python python/export_table_transformer_ane.py --batch_size {b}"
        stdout, stderr = run_command(export_cmd)
        if "Parity test passed!" not in stdout:
            print(f"Export failed or parity check failed for batch size {b}")
            print(stdout)
            print(stderr)
            # continue # Try to proceed anyway if it's just a warning

        # 2. Benchmark FP32
        fp32_model = f"models/table-transformer-structure-recognition-ane-b{b}.onnx"
        bench_cmd = f"python python/benchmark_ane.py --model {fp32_model} --iterations {args.iterations}"
        stdout, stderr = run_command(bench_cmd)
        
        # Parse latency and FPS from benchmark output
        # Latency: 16.31 ms | Throughput: 61.30 FPS
        import re
        latency_match = re.search(r"Latency:\s+([\d.]+)\s+ms", stdout)
        fps_match = re.search(r"Throughput:\s+([\d.]+)\s+FPS", stdout)
        
        if latency_match and fps_match:
            latency = float(latency_match.group(1))
            fps = float(fps_match.group(1))
            results.append({
                "Batch": b,
                "Type": "FP32",
                "Latency (ms)": latency,
                "Latency/Item (ms)": latency / b,
                "Throughput (FPS)": fps,
                "Total FPS": fps * b
            })
        
        # 3. Benchmark FP16
        fp16_model = f"models/table-transformer-structure-recognition-ane-b{b}_fp16.onnx"
        if os.path.exists(fp16_model):
            bench_cmd = f"python python/benchmark_ane.py --model {fp16_model} --iterations {args.iterations}"
            stdout, stderr = run_command(bench_cmd)
            
            latency_match = re.search(r"Latency:\s+([\d.]+)\s+ms", stdout)
            fps_match = re.search(r"Throughput:\s+([\d.]+)\s+FPS", stdout)
            
            if latency_match and fps_match:
                latency = float(latency_match.group(1))
                fps = float(fps_match.group(1))
                results.append({
                    "Batch": b,
                    "Type": "FP16",
                    "Latency (ms)": latency,
                    "Latency/Item (ms)": latency / b,
                    "Throughput (FPS)": fps,
                    "Total FPS": fps * b
                })

    print("\n" + "="*80)
    print(f"{'Batch':<8} | {'Type':<6} | {'Latency (ms)':<15} | {'Lat/Item (ms)':<15} | {'FPS':<10} | {'Total FPS':<10}")
    print("-" * 80)
    for res in results:
        print(f"{res['Batch']:<8} | {res['Type']:<6} | {res['Latency (ms)']:<15.2f} | {res['Latency/Item (ms)']:<15.2f} | {res['Throughput (FPS)']:<10.2f} | {res['Total FPS']:<10.2f}")
    print("="*80)
    
    # Save to CSV
    import csv
    with open("ane_benchmark_results.csv", "w", newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print("Results saved to ane_benchmark_results.csv")

if __name__ == "__main__":
    main()
