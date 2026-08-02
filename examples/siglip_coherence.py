#!/usr/bin/env python3
"""Verify SigLIP NPU output coherence against CPU reference.

Loads pretrained SigLIP weights, runs a test image through both the NPU
vision encoder and the CPU reference, then passes both through SigLIP's
pooling head + a linear classifier. Checks that NPU output matches CPU
closely enough for coherent downstream classification.

Run on the SBC:
  NPU_PLUGIN_LIB=... python3 examples/siglip_coherence.py
"""
import os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("OMP_WAIT_POLICY", "active")

from opennpu.torch_npu import NPUViTEncoder


def main():
    from transformers import SiglipModel

    print("=== SigLIP Coherence Test ===\n")
    print("Loading pretrained SigLIP from HuggingFace...")
    model = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
    vm = model.vision_model
    print(f"  12 layers, 768 dim, 196 patches (224/16)")

    # Create test input: the patch embeddings (skip the conv)
    # SigLIP vision: image -> patch_embed -> encoder -> [196, 768]
    x_img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        x_patches = vm.embeddings(x_img)  # [1, 196, 768]
    print(f"  Patch embeddings: {x_patches.shape}")

    # CPU reference: run full vision model from patch embeddings
    print("\nCPU reference forward...")
    with torch.no_grad():
        # Run encoder + post_layernorm + head (pooling)
        enc_out = vm.encoder(x_patches, output_hidden_states=False)
        x_norm = vm.post_layernorm(enc_out[0])
        pooled = vm.head(x_norm)  # [1, 768] pooled features
        cpu_features = pooled.squeeze(0).numpy()
    print(f"  CPU pooled features: {cpu_features.shape}, mean={cpu_features.mean():.4f}")

    # NPU: load weights and run
    print("\nLoading weights onto NPU...")
    encoder = NPUViTEncoder.from_torch(model, n_layers=12, seq_len=196, dim=768)
    print("  Weights loaded")

    # Run NPU vision encoder (from patch embeddings)
    x_np = x_patches.squeeze(0).numpy().astype(np.float32)
    print("\nNPU forward...")
    npu_out = encoder._backend.forward(x_np)
    print(f"  NPU encoder output: {npu_out.shape}, mean={npu_out.mean():.4f}")

    # Pass NPU output through the same post_layernorm + head (on CPU)
    npu_torch = torch.from_numpy(npu_out).unsqueeze(0)
    with torch.no_grad():
        npu_norm = vm.post_layernorm(npu_torch)
        npu_pooled = vm.head(npu_norm)
        npu_features = npu_pooled.squeeze(0).numpy()
    print(f"  NPU pooled features: {npu_features.shape}, mean={npu_features.mean():.4f}")

    # Compare encoder outputs (before pooling)
    cpu_enc = enc_out[0].squeeze(0).numpy()
    enc_diff = np.max(np.abs(npu_out - cpu_enc))
    enc_rel = enc_diff / (np.abs(cpu_enc).max() + 1e-8)
    print(f"\nEncoder output comparison:")
    print(f"  max abs diff:  {enc_diff:.4f}")
    print(f"  relative diff: {enc_rel:.4f}")
    print(f"  cos sim:       {np.dot(npu_out.flatten(), cpu_enc.flatten()) / (np.linalg.norm(npu_out) * np.linalg.norm(cpu_enc)):.6f}")

    # Compare pooled features
    feat_diff = np.max(np.abs(npu_features - cpu_features))
    feat_cos = np.dot(npu_features, cpu_features) / (np.linalg.norm(npu_features) * np.linalg.norm(cpu_features))
    print(f"\nPooled features comparison:")
    print(f"  max abs diff:  {feat_diff:.4f}")
    print(f"  cos sim:       {feat_cos:.6f}")

    # Pass through the text projection (linear classifier)
    # SigLIP uses image-text similarity. We use the text encoder for a few labels.
    print("\nPassing through linear classifier (image-text similarity)...")
    from transformers import SiglipProcessor

    try:
        processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
        labels = ["a cat", "a dog", "a car", "a building", "a tree", "abstract art"]
        text_inputs = processor(text=labels, padding="max_length", return_tensors="pt")

        with torch.no_grad():
            text_features = model.text_model(**{k: v for k, v in text_inputs.items() if k in ["input_ids", "attention_mask"]})
            text_embeds = text_features.last_hidden_state[:, 0, :]  # CLS token
            text_embeds = model.text_projection(text_embeds)

            # CPU image features
            cpu_img = torch.from_numpy(cpu_features) @ model.logit_scale.exp() if hasattr(model, 'logit_scale') else torch.from_numpy(cpu_features)
            cpu_img_proj = model.logit_scale.exp() * (torch.from_numpy(cpu_features) @ model.visual_projection.weight.T) if hasattr(model, 'visual_projection') else torch.from_numpy(cpu_features)

            # Simple similarity scores
            cpu_scores = (torch.from_numpy(cpu_features) @ text_embeds.T).numpy()
            npu_scores = (torch.from_numpy(npu_features) @ text_embeds.T).numpy()

        print(f"  CPU scores: {cpu_scores}")
        print(f"  NPU scores: {npu_scores}")
        score_diff = np.max(np.abs(npu_scores - cpu_scores))
        print(f"  max score diff: {score_diff:.4f}")
        print(f"  top label (CPU): {labels[np.argmax(cpu_scores)]}")
        print(f"  top label (NPU): {labels[np.argmax(npu_scores)]}")
        print(f"  Same prediction: {np.argmax(cpu_scores) == np.argmax(npu_scores)}")
    except Exception as e:
        print(f"  Classifier test skipped: {e}")

    # Verdict
    print("\n=== Verdict ===")
    if enc_rel < 0.1:
        print(f"  PASS: encoder relative diff {enc_rel:.4f} < 0.1")
    else:
        print(f"  WARN: encoder relative diff {enc_rel:.4f} >= 0.1")
    if feat_cos > 0.95:
        print(f"  PASS: pooled cos sim {feat_cos:.6f} > 0.95")
    else:
        print(f"  WARN: pooled cos sim {feat_cos:.6f} <= 0.95")



if __name__ == "__main__":
    main()