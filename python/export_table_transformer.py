import torch
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
import onnx
import onnxruntime
import os

# Define paths
MODEL_NAME = "microsoft/table-transformer-structure-recognition"
OUTPUT_DIR = "../models"
ONNX_PATH = os.path.join(OUTPUT_DIR, "table-transformer-structure-recognition.onnx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading model: {MODEL_NAME}")
model = TableTransformerForObjectDetection.from_pretrained(MODEL_NAME)
processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
model.eval()

# Create dummy input
# The model expects pixel_values of shape [batch_size, 3, height, width]
# and pixel_mask of shape [batch_size, height, width]
# Standard size for this model is often 800-1000, let's use 1000x1000
dummy_input = torch.randn(1, 3, 1000, 1000)
# mask is optional in some exports but good to check. For standard export we usually just pass pixel_values

print("Exporting to ONNX...")
# Dynamic axes are crucial for variable image sizes
dynamic_axes = {
    "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
    "logits": {0: "batch_size", 1: "num_queries"},
    "pred_boxes": {0: "batch_size", 1: "num_queries"},
}

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["pixel_values"],
    output_names=["logits", "pred_boxes"],
    dynamic_axes=dynamic_axes,
)

print(f"Model exported to {ONNX_PATH}")

# Verify
print("Verifying ONNX model...")
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("ONNX model verified.")

# Quantization / FP16 (Optional but recommended for size)
print("Converting to FP16...")
from onnxconverter_common import float16
model_fp16 = float16.convert_float_to_float16(onnx_model)
ONNX_FP16_PATH = ONNX_PATH.replace(".onnx", "_fp16.onnx")
onnx.save(model_fp16, ONNX_FP16_PATH)
print(f"FP16 Model saved to {ONNX_FP16_PATH}")
