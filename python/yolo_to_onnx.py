import torch
import os
from ultralytics import YOLO

# Load the YOLO11 model

model = YOLO("../models/yolov8s-doclaynet.pt")

model_path = "../models/yolov8s-doclaynet.pt"
model = YOLO(model_path)

# Define the model name based on the path
model_name = os.path.splitext(os.path.basename(model_path))[0]

# Export the model to ONNX format with different batch sizes
for batch_size in [2, 4, 8, 32]:
    onnx_file = model.export(format="onnx", batch=batch_size, simplify=True)
    
    # Construct the new file name
    new_file_name = f"{model_name}_batchsize{batch_size}.onnx"
    
    # Rename the exported ONNX file
    if os.path.exists(onnx_file):
        os.rename(onnx_file, new_file_name)
        print(f"Renamed {onnx_file} to {new_file_name}")
    else:
        print(f"Exported file {onnx_file} does not exist.")
