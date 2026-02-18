import torch
from transformers import TableTransformerForObjectDetection
import onnx
import os
import argparse

MODEL_NAME = "microsoft/table-transformer-structure-recognition"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    
    batch_size = args.batch_size
    output_path = os.path.join(OUTPUT_DIR, f"table-transformer-fixed-b{batch_size}.onnx")
    
    print(f"Loading model: {MODEL_NAME}")
    model = TableTransformerForObjectDetection.from_pretrained(MODEL_NAME)
    model.eval()
    
    dummy_input = torch.randn(batch_size, 3, 1000, 1000)
    pixel_mask = torch.ones((batch_size, 1000, 1000))
    
    print(f"Exporting to ONNX with batch size {batch_size}...")
    torch.onnx.export(
        model,
        (dummy_input, pixel_mask),
        output_path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["pixel_values", "pixel_mask"],
        output_names=["logits", "pred_boxes"],
    )
    print(f"Exported to {output_path}")

if __name__ == "__main__":
    main()
