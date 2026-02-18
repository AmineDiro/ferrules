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
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (x - mean) / std
        return y * self.weight + self.bias

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
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, **kwargs):
        B, C, _, S_tgt = query.shape
        _, _, _, S_src = key.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        mh_q = q.split(self.head_dim, dim=1)
        mh_k = k.transpose(1, 3).split(self.head_dim, dim=3)
        mh_v = v.split(self.head_dim, dim=1)

        heads = []
        for i in range(self.num_heads):
            # attn_weights = torch.einsum('bchq,bkhc->bkhq', mh_q[i], mh_k[i]) * self.scale
            # bchq: (B, C, 1, S), bkhc: (B, S, 1, C) -> bkhq: (B, S, 1, S)
            q = mh_q[i].squeeze(2) # (B, C, S)
            k = mh_k[i].squeeze(2) # (B, S, C)
            attn_weights = torch.matmul(k, q).unsqueeze(2) * self.scale

            if attn_mask is not None:
                attn_weights = attn_weights + attn_mask
                
            attn_weights = F.softmax(attn_weights, dim=1)
            attn_weights = self.dropout(attn_weights)
            
            # head = torch.einsum('bchs,bshq->bchq', mh_v[i], attn_weights)
            # bchs: (B, C, 1, S), bshq: (B, S, 1, S) -> bchq: (B, C, 1, S)
            v = mh_v[i].squeeze(2) # (B, C, S)
            attn = attn_weights.squeeze(2) # (B, S, S)
            head = torch.matmul(v, attn).unsqueeze(2)
            heads.append(head)

        out = torch.cat(heads, dim=1)
        context_layer = self.out_proj(out)
        
        return context_layer, None

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
                new_ln.weight.data = child.weight.data.view(1, -1, 1, 1)
                new_ln.bias.data = child.bias.data.view(1, -1, 1, 1)
                setattr(module, name, new_ln)
            else:
                self._patch_internals(child)

    def _extract_tensor(self, x):
        if hasattr(x, 'tensors'):
            return x.tensors
        if isinstance(x, (list, tuple)):
             return self._extract_tensor(x[-1])
        return x

    def forward(self, pixel_values):
        B, C, H, W = pixel_values.shape
        pixel_mask = torch.ones((B, H, W), device=pixel_values.device)
        
        # 1. Backbone
        backbone_outputs = self.backbone(pixel_values, pixel_mask)
        target_features = None
        if hasattr(backbone_outputs, 'feature_maps'):
            for fm in backbone_outputs.feature_maps:
                t = fm[0] if isinstance(fm, (list, tuple)) else fm
                if hasattr(t, 'tensors'): t = t.tensors
                if t.shape[1] == 512:
                    target_features = t
                    break
        if target_features is None:
             def extract_512(x):
                 if hasattr(x, 'tensors'): x = x.tensors
                 if isinstance(x, torch.Tensor) and x.shape[1] == 512: return x
                 if isinstance(x, (list, tuple)):
                     for item in x:
                         res = extract_512(item)
                         if res is not None: return res
                 return None
             target_features = extract_512(backbone_outputs)
        features = target_features

        # 2. Input Projection
        projected_features = self.input_projection(features)
        B, D, H_feat, W_feat = projected_features.shape
        src = projected_features.view(B, D, 1, H_feat*W_feat)
        
        # 3. Position Embeddings
        feat_mask = torch.ones((B, H_feat, W_feat), device=pixel_values.device)
        pos_outputs = self.backbone.position_embedding(projected_features, feat_mask)
        pos = self._extract_tensor(pos_outputs).view(B, D, 1, H_feat*W_feat)
        
        # 4. Encoder
        memory = src
        for layer in self.encoder.layers:
            residual = memory
            hidden_states = layer.self_attn_layer_norm(memory)
            q = hidden_states + pos
            k = hidden_states + pos
            v = hidden_states
            attn_output, _ = layer.self_attn(query=q, key=k, value=v)
            hidden_states = residual + attn_output
            
            residual = hidden_states
            hidden_states = layer.final_layer_norm(hidden_states)
            ffn_output = layer.fc1(hidden_states)
            ffn_output = layer.activation_fn(ffn_output)
            ffn_output = layer.fc2(ffn_output)
            memory = residual + ffn_output
            
        memory = self.encoder.layernorm(memory)

        # 5. Decoder
        query_pos = self.query_position_embeddings.weight.T.unsqueeze(0).unsqueeze(2).expand(B, -1, -1, -1)
        tgt = torch.zeros_like(query_pos)
        
        for layer in self.decoder.layers:
            # Self Attention
            residual = tgt
            hidden_states = layer.self_attn_layer_norm(tgt)
            q = hidden_states + query_pos
            k = hidden_states + query_pos
            v = hidden_states
            attn_output, _ = layer.self_attn(query=q, key=k, value=v)
            tgt = residual + attn_output
            
            # Encoder Attention (Cross)
            residual = tgt
            hidden_states = layer.encoder_attn_layer_norm(tgt)
            q = hidden_states + query_pos
            k = memory + pos
            v = memory
            attn_output, _ = layer.encoder_attn(query=q, key=k, value=v)
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

def main():
    print(f"Loading model: {MODEL_NAME}")
    original_hf_model = TableTransformerForObjectDetection.from_pretrained(MODEL_NAME)
    original_hf_model.eval()
    
    print("Creating ANE optimized model...")
    ane_model = TableTransformerANE(original_hf_model)
    ane_model.eval()
    
    print("Running parity test...")
    dummy_input = torch.randn(1, 3, 1000, 1000)
    pixel_mask = torch.ones((1, 1000, 1000))
    
    with torch.no_grad():
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

    print("Exporting to ONNX...")
    torch.onnx.export(
        ane_model,
        dummy_input,
        ANE_ONNX_PATH,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "pred_boxes": {0: "batch_size"},
        },
    )
    print(f"ANE Model exported to {ANE_ONNX_PATH}")

    print("Converting to FP16...")
    try:
        from onnxconverter_common import float16
        onnx_model = onnx.load(ANE_ONNX_PATH)
        model_fp16 = float16.convert_float_to_float16(onnx_model)
        onnx.save(model_fp16, ANE_FP16_ONNX_PATH)
        print(f"FP16 ANE Model saved to {ANE_FP16_ONNX_PATH}")
    except ImportError:
        print("onnxconverter_common not found, skipping FP16 conversion.")

if __name__ == "__main__":
    main()
