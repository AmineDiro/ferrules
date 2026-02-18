import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
import onnx
import os
import copy
from typing import Optional

# Define paths
MODEL_NAME = "microsoft/table-transformer-structure-recognition"
OUTPUT_DIR = "models"
ANE_ONNX_PATH = os.path.join(OUTPUT_DIR, "table-transformer-structure-recognition-ane.onnx")
ANE_FP16_ONNX_PATH = os.path.join(OUTPUT_DIR, "table-transformer-structure-recognition-ane_fp16.onnx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

class LayerNormANE(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(1, num_channels, eps=eps)

    def forward(self, x):
        # x is (B, C, 1, S) or similar rank-4
        # GroupNorm(1, C) is equivalent to LayerNorm over C, H, W
        return self.gn(x)

class ANEOptimizedAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, query, key, value, attn_mask=None, **kwargs):
        # query, key, value are (B, C, 1, S)
        B, C, _, S_tgt = query.shape
        _, _, _, S_src = key.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Keep everything 4D for ANE
        # qn: (BN, 1, head_dim, S_tgt)
        # kn: (BN, 1, head_dim, S_src)
        # vn: (BN, 1, head_dim, S_src)
        qn = q.reshape(B * self.num_heads, 1, self.head_dim, S_tgt)
        kn = k.reshape(B * self.num_heads, 1, self.head_dim, S_src)
        vn = v.reshape(B * self.num_heads, 1, self.head_dim, S_src)

        # (BN, 1, S_src, head_dim) @ (BN, 1, head_dim, S_tgt) -> (BN, 1, S_src, S_tgt)
        aw = torch.matmul(kn.transpose(2, 3), qn) * self.scale

        if attn_mask is not None:
            # attn_mask: (B, S_src, 1, S_tgt) -> (B*H, 1, S_src, S_tgt)
            m = attn_mask.reshape(B, 1, S_src, S_tgt).expand(B * self.num_heads, 1, S_src, S_tgt)
            aw = aw + m
                
        aw = F.softmax(aw, dim=2) # Normalize across S_src (dim 2)
        aw = self.dropout(aw)
        
        # (BN, 1, head_dim, S_src) @ (BN, 1, S_src, S_tgt) -> (BN, 1, head_dim, S_tgt)
        h = torch.matmul(vn, aw)
        
        # (BN, 1, head_dim, S_tgt) -> (B, C, 1, S_tgt)
        out = h.reshape(B, C, 1, S_tgt)
        return self.out_proj(out), None

class TableTransformerANE(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        model_copy = copy.deepcopy(original_model)
        
        self.backbone = model_copy.model.backbone
        self.input_projection = model_copy.model.input_projection
        self.query_position_embeddings = model_copy.model.query_position_embeddings
        self.encoder = model_copy.model.encoder
        self.decoder = model_copy.model.decoder
        
        self.class_labels_classifier = model_copy.class_labels_classifier
        self.bbox_predictor = model_copy.bbox_predictor
        
        self._patch_internals(self.encoder)
        self._patch_internals(self.decoder)
        self._patch_internals(self.bbox_predictor)
        self._patch_classifier()
        
        # Precompute positions for 1000x1000 input
        print("Precomputing positional embeddings...")
        with torch.no_grad():
            dummy_x = torch.zeros(1, 3, 1000, 1000)
            dummy_m = torch.ones(1, 1000, 1000)
            backbone_out = self.backbone(dummy_x, dummy_m)
            # Find the 512-channel feature map
            def find_512(x):
                if isinstance(x, torch.Tensor) and x.dim() == 4 and x.shape[1] == 512:
                    return x
                if hasattr(x, 'tensors'): return find_512(x.tensors)
                if isinstance(x, (list, tuple)):
                    for item in x:
                        res = find_512(item)
                        if res is not None: return res
                return None
            
            feat = find_512(backbone_out)
            proj_feat = self.input_projection(feat)
            _, D, H_feat, W_feat = proj_feat.shape
            
            feat_mask = torch.ones((1, H_feat, W_feat))
            pos_out = self.backbone.position_embedding(proj_feat, feat_mask)
            if hasattr(pos_out, 'tensors'): pos_out = pos_out.tensors
            elif isinstance(pos_out, (list, tuple)): pos_out = pos_out[-1]
            
            self.register_buffer("fixed_pos", pos_out.view(1, D, 1, H_feat * W_feat))
            self.register_buffer("fixed_query_pos", self.query_position_embeddings.weight.T.unsqueeze(0).unsqueeze(2))
            self.H_feat = H_feat
            self.W_feat = W_feat

    def _patch_classifier(self):
        child = self.class_labels_classifier
        new_conv = nn.Conv2d(child.in_features, child.out_features, 1)
        new_conv.weight.data = child.weight.data.view(new_conv.weight.shape)
        new_conv.bias.data = child.bias.data
        self.class_labels_classifier = new_conv

    def _patch_internals(self, module):
        for name, child in module.named_children():
            if "Attention" in child.__class__.__name__:
                new_attn = ANEOptimizedAttention(child.embed_dim, child.num_heads, child.dropout if hasattr(child, 'dropout') else 0.0)
                new_attn.q_proj.weight.data = child.q_proj.weight.data.view(new_attn.q_proj.weight.shape)
                new_attn.q_proj.bias.data = child.q_proj.bias.data
                new_attn.k_proj.weight.data = child.k_proj.weight.data.view(new_attn.k_proj.weight.shape)
                new_attn.k_proj.bias.data = child.k_proj.bias.data
                new_attn.v_proj.weight.data = child.v_proj.weight.data.view(new_attn.v_proj.weight.shape)
                new_attn.v_proj.bias.data = child.v_proj.bias.data
                new_attn.out_proj.weight.data = child.out_proj.weight.data.view(new_attn.out_proj.weight.shape)
                new_attn.out_proj.bias.data = child.out_proj.bias.data
                setattr(module, name, new_attn)
            elif isinstance(child, nn.Linear):
                new_conv = nn.Conv2d(child.in_features, child.out_features, 1, bias=child.bias is not None)
                new_conv.weight.data = child.weight.data.view(new_conv.weight.shape)
                if child.bias is not None:
                    new_conv.bias.data = child.bias.data
                setattr(module, name, new_conv)
            elif isinstance(child, nn.LayerNorm):
                new_ln = LayerNormANE(child.normalized_shape[0], eps=child.eps)
                new_ln.gn.weight.data = child.weight.data
                new_ln.gn.bias.data = child.bias.data
                setattr(module, name, new_ln)
            elif "FrozenBatchNorm" in child.__class__.__name__:
                new_bn = nn.BatchNorm2d(child.weight.shape[0], eps=child.eps)
                new_bn.weight.data = child.weight.data
                new_bn.bias.data = child.bias.data
                new_bn.running_mean.data = child.running_mean.data
                new_bn.running_var.data = child.running_var.data
                new_bn.eps = child.eps
                setattr(module, name, new_bn)
            else:
                self._patch_internals(child)

    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        
        # 1. Backbone - Use fixed ones mask to satisfy the API if needed, 
        # but the actual feature extraction is what we want.
        # We call the model directly to avoid extra wrappers.
        pixel_mask = torch.ones((B, 1000, 1000), device=pixel_values.device)
        backbone_outputs = self.backbone(pixel_values, pixel_mask)
        
        # Robustly extract the 512-channel feature map
        def find_512(x):
            if isinstance(x, torch.Tensor) and x.dim() == 4 and x.shape[1] == 512:
                return x
            if hasattr(x, 'tensors'): return find_512(x.tensors)
            if isinstance(x, (list, tuple)):
                for item in x:
                    res = find_512(item)
                    if res is not None: return res
            return None
            
        features = find_512(backbone_outputs)
        
        # 2. Input Projection
        projected_features = self.input_projection(features)
        S = self.H_feat * self.W_feat
        D = projected_features.shape[1]
        src = projected_features.view(B, D, 1, S)
        
        # 3. Use Precomputed Position Embeddings
        pos = self.fixed_pos.expand(B, -1, -1, -1)
        
        # 4. Encoder
        memory = src
        for layer in self.encoder.layers:
            residual = memory
            hidden_states = layer.self_attn_layer_norm(memory)
            q = hidden_states + pos
            attn_output, _ = layer.self_attn(query=q, key=q, value=hidden_states)
            hidden_states = residual + attn_output
            
            residual = hidden_states
            hidden_states = layer.final_layer_norm(hidden_states)
            ffn_output = layer.fc1(hidden_states)
            ffn_output = layer.activation_fn(ffn_output)
            ffn_output = layer.fc2(ffn_output)
            memory = residual + ffn_output
            
        memory = self.encoder.layernorm(memory)

        # 5. Decoder
        query_pos = self.fixed_query_pos.expand(B, -1, -1, -1)
        tgt = torch.zeros_like(query_pos)
        
        for layer in self.decoder.layers:
            # Self Attention
            residual = tgt
            hidden_states = layer.self_attn_layer_norm(tgt)
            q = hidden_states + query_pos
            attn_output, _ = layer.self_attn(query=q, key=q, value=hidden_states)
            tgt = residual + attn_output
            
            # Encoder Attention (Cross)
            residual = tgt
            hidden_states = layer.encoder_attn_layer_norm(tgt)
            q = hidden_states + query_pos
            k = memory + pos
            attn_output, _ = layer.encoder_attn(query=q, key=k, value=memory)
            tgt = residual + attn_output
            
            # FFN
            residual = tgt
            hidden_states = layer.final_layer_norm(tgt)
            ffn_output = layer.fc1(hidden_states)
            ffn_output = layer.activation_fn(ffn_output)
            ffn_output = layer.fc2(ffn_output)
            tgt = residual + ffn_output
            
        hs = self.decoder.layernorm(tgt)

        # 6. Heads
        logits = self.class_labels_classifier(hs).squeeze(2).transpose(1, 2)
        
        boxes_hs = hs
        for i, layer in enumerate(self.bbox_predictor.layers):
            boxes_hs = layer(boxes_hs)
            if i < len(self.bbox_predictor.layers) - 1:
                boxes_hs = F.relu(boxes_hs)
        
        pred_boxes = boxes_hs.squeeze(2).transpose(1, 2)
        
        from collections import namedtuple
        DETROutput = namedtuple('DETROutput', ['logits', 'pred_boxes'])
        return DETROutput(logits=logits, pred_boxes=torch.sigmoid(pred_boxes))

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the exported model")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="HuggingFace model name")
    parser.add_argument("--output_name", type=str, default=None, help="Output base name")
    args = parser.parse_args()

    batch_size = args.batch_size
    model_id = args.model_name
    
    if args.output_name:
        base_output = args.output_name
    else:
        # e.g., table-transformer-structure-recognition-ane-b1
        base_output = f"{model_id.split('/')[-1]}-ane-b{batch_size}"
    
    output_path = os.path.join(OUTPUT_DIR, f"{base_output}.onnx")
    output_fp16_path = os.path.join(OUTPUT_DIR, f"{base_output}_fp16.onnx")

    print(f"Loading model: {model_id}")
    original_hf_model = TableTransformerForObjectDetection.from_pretrained(model_id)
    original_hf_model.eval()
    
    print(f"Creating ANE optimized model for batch size {batch_size}...")
    ane_model = TableTransformerANE(original_hf_model)
    ane_model.eval()
    
    print("Running parity test...")
    dummy_input = torch.randn(batch_size, 3, 1000, 1000)
    pixel_mask = torch.ones((batch_size, 1000, 1000))
    
    with torch.no_grad():
        # original model might need special padding if not perfectly handled, 
        # but let's assume it works for the parity check with dummy input
        original_output = original_hf_model(dummy_input, pixel_mask=pixel_mask)
        ane_output = ane_model(dummy_input)
        
    logits_diff = torch.abs(original_output.logits - ane_output.logits).max().item()
    print(f"Max Logits Difference: {logits_diff}")
    boxes_diff = torch.abs(original_output.pred_boxes - ane_output.pred_boxes).max().item()
    print(f"Max Boxes Difference: {boxes_diff}")
    
    if logits_diff < 0.1 and boxes_diff < 0.1:
        print("✅ Parity test passed!")
    else:
        print(f"⚠️ Parity test warning: Differences too large (logits={logits_diff}, boxes={boxes_diff})")

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        ane_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        # ANE prefers static shapes, so we do NOT use dynamic_axes here for batch_size
    )
    print(f"ANE Model exported to {output_path}")

    print("Converting to FP16...")
    try:
        from onnxconverter_common import float16
        onnx_model = onnx.load(output_path)
        model_fp16 = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
        onnx.save(model_fp16, output_fp16_path)
        print(f"FP16 ANE Model saved to {output_fp16_path}")
    except ImportError:
        print("onnxconverter_common not found, skipping FP16 conversion.")

if __name__ == "__main__":
    main()
