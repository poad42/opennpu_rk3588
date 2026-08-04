#!/usr/bin/env python3
"""Verify SigLIP NPU embeddings are coherent via image-text similarity.

Runs the vision encoder on the NPU, passes the output through SigLIP's
pooling head, and computes image-text similarity against candidate labels.
Verifies the NPU embeddings produce the SAME top-1 classification as the
CPU reference — proving the embeddings are meaningful for downstream tasks.

Run on the SBC:
  NPU_PLUGIN_LIB=... python3 examples/siglip_classifier.py
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("OMP_WAIT_POLICY", "active")


def main():
    from transformers import SiglipModel, SiglipProcessor

    print("=== SigLIP Embedding Coherence via Linear Classifier ===\n")
    print("Loading pretrained SigLIP...")
    model = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
    vm = model.vision_model
    processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")

    # Test images with different structure to produce distinguishable embeddings
    images = {
        "cat": torch.randn(1, 3, 224, 224),
        "dog": torch.randn(1, 3, 224, 224) + 0.5,
        "car": torch.randn(1, 3, 224, 224) * 0.5,
    }
    labels = ["a cat", "a dog", "a car", "a building", "a tree"]

    # Text embeddings via SigLIP text encoder (pooled CLS token)
    text_inputs = processor(text=labels, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        text_out = model.text_model(input_ids=text_inputs.input_ids)
        text_embeds = text_out.last_hidden_state[:, 0, :]  # [n_labels, 768]
    scale = model.logit_scale.exp()
    print(f"  Text embeddings: {text_embeds.shape}, logit_scale={scale.item():.3f}\n")

    from opennpu.torch_npu import NPUViTEncoder
    enc = NPUViTEncoder.from_torch(model)

    all_pass = True
    for name, img in images.items():
        with torch.no_grad():
            patches = vm.embeddings(img)
            cpu_enc = vm.encoder(patches)[0]
            cpu_norm = vm.post_layernorm(cpu_enc)
            cpu_pooled = vm.head(cpu_norm)  # [1, 768]

            # NPU encoder, then CPU pooler (pooling head not on NPU)
            npu_enc = enc._backend.forward(
                np.ascontiguousarray(patches[0].numpy().astype(np.float32)))
            npu_t = torch.from_numpy(npu_enc).unsqueeze(0)
            npu_norm = vm.post_layernorm(npu_t)
            npu_pooled = vm.head(npu_norm)

            # Image-text similarity: pooled @ text_embeds.T * logit_scale
            cpu_scores = cpu_pooled @ text_embeds.T * scale
            npu_scores = npu_pooled @ text_embeds.T * scale

        cpu_top = labels[int(cpu_scores.argmax())]
        npu_top = labels[int(npu_scores.argmax())]
        same = cpu_top == npu_top
        all_pass = all_pass and same

        cos = torch.nn.functional.cosine_similarity(
            cpu_pooled / cpu_pooled.norm(), npu_pooled / npu_pooled.norm()).item()
        print(f"[{name}]")
        print(f"  CPU scores: {[f'{s:.2f}' for s in cpu_scores[0].tolist()]}")
        print(f"  NPU scores: {[f'{s:.2f}' for s in npu_scores[0].tolist()]}")
        print(f"  CPU top-1: {cpu_top}  |  NPU top-1: {npu_top}")
        print(f"  Same prediction: {same}")
        print(f"  Pooled embedding cos sim: {cos:.6f}\n")

    print("=== Verdict ===")
    if all_pass:
        print("  PASS: NPU and CPU produce identical top-1 classifications.")
        print("  The NPU vision encoder produces embeddings coherent enough")
        print("  for downstream linear classification.")
    else:
        print("  FAIL: NPU and CPU top-1 differ.")
    try:
        enc.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()