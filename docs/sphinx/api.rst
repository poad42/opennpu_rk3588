API Reference
=============

Python Modules
--------------

.. autosummary::
   :toctree: generated

   opennpu.torch_npu
   opennpu.jax_npu_vision
   opennpu.lm
   opennpu.runtime
   opennpu.onnx_runner

C Backend
---------

The C backend is compiled into ``libpjrt_npu.so`` and accessed via ctypes.

Key exported functions:

- ``npu_cna_cache_setup(n_w, M, max_K, max_N)`` — initialize CNA matmul engine
- ``npu_cna_cache_load(w_idx, W, K, N)`` — load weight onto NPU
- ``npu_cna_cache_run_m(w_idx, M, X, Z)`` — run matmul on NPU
- ``npu_vision_init(dim, n_heads, inter, max_seq, max_layers, eps)`` — init vision encoder
- ``npu_vision_forward(input, seq_len, n_layers, output)`` — run ViT encoder
- ``npu_lm_init(dim, n_heads, inter, vocab, n_layers, max_seq, eps)`` — init transformer decoder
- ``npu_lm_step(token_id, logits)`` — single-token decode with KV cache
- ``npu_lm_prefill(ids, n_ids, logits)`` — process prompt tokens
