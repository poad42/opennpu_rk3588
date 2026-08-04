"""JAX integration for OpenNPU vision encoder.

Provides npu_vit_forward as a JAX-compatible function that runs the
ViT encoder on the NPU. Accepts jnp.ndarray, returns jnp.ndarray.

Usage:
    from opennpu.jax_npu_vision import load_random_weights, npu_vit_forward
    load_random_weights(n_layers=12)
    out = npu_vit_forward(x)  # jnp.ndarray in, jnp.ndarray out
"""
import os, ctypes
import numpy as np
import jax.numpy as jnp
from opennpu.torch_npu import _NPUBackend

_PLUGIN_PATH = os.environ.get("NPU_PLUGIN_LIB",
    os.path.join(os.path.dirname(__file__), "pjrt_c", "libpjrt_npu.so"))

_backend = None

def _np(x):
    if hasattr(x, 'detach'): return x.detach().cpu().numpy()
    return np.asarray(x)

def load_vit_weights(model, n_layers=12):
    global _backend
    _backend = _NPUBackend.get()
    if hasattr(model, 'vision_model'):
        vm = model.vision_model
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        vm = model
    elif hasattr(model, 'blocks'):
        vm = model
    elif isinstance(model, list):
        _backend.init_vision()
        for i, (wq, wp, wf, wd, l1w, l1b, l2w, l2b) in enumerate(model[:n_layers]):
            _backend.load_layer(i, _np(wq), _np(wp), _np(wf), _np(wd),
                                _np(l1w), _np(l1b), _np(l2w), _np(l2b))
        return
    else:
        raise ValueError("Expected SiglipModel, ViTModel, or list")
    cfg = getattr(vm, 'config', None)
    dim = getattr(cfg, 'hidden_size', 768)
    n_heads = getattr(cfg, 'num_attention_heads', 12)
    inter = getattr(cfg, 'intermediate_size', 3072)
    if cfg and hasattr(cfg, 'image_size') and hasattr(cfg, 'patch_size'):
        seq_len = (cfg.image_size // cfg.patch_size) ** 2
    else:
        seq_len = 197
    _backend.init_vision(dim, n_heads, inter, max(seq_len, 256), n_layers)
    if hasattr(model, 'vision_model'):
        vm = model.vision_model
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        vm = model
    elif hasattr(model, 'blocks'):
        vm = model
    elif isinstance(model, list):
        for i, (wq, wp, wf, wd, l1w, l1b, l2w, l2b) in enumerate(model[:n_layers]):
            _backend.load_layer(i, _np(wq), _np(wp), _np(wf), _np(wd),
                                _np(l1w), _np(l1b), _np(l2w), _np(l2b))
        return
    else:
        raise ValueError("Expected SiglipModel, ViTModel, or list")
    if hasattr(vm, 'encoder') and hasattr(vm.encoder, 'layers'):
        layers = vm.encoder.layers
    elif hasattr(vm, 'encoder') and hasattr(vm.encoder, 'layer'):
        layers = vm.encoder.layer
    elif hasattr(vm, 'blocks'):
        layers = vm.blocks
    else:
        raise ValueError("Cannot find encoder layers")
    from opennpu.torch_npu import NPUViTEncoder
    enc = NPUViTEncoder(n_layers=n_layers, seq_len=seq_len, dim=dim,
                        inter=inter, n_heads=n_heads)
    enc._backend = _backend
    for i, block in enumerate(layers[:n_layers]):
        enc._load_block_weights(i, block)
    enc._weights_loaded = True

def load_random_weights(n_layers=12, dim=768, inter=3072, seed=42, n_heads=12, seq_len=197):
    global _backend
    _backend = _NPUBackend.get()
    _backend.init_vision(dim, n_heads, inter, max(seq_len, 256), n_layers)
    rng = np.random.RandomState(seed)
    for i in range(n_layers):
        wq = (rng.randn(dim, dim*3) * 0.02).astype(np.float16)
        wp = (rng.randn(dim, dim) * 0.02).astype(np.float16)
        wf = (rng.randn(dim, inter) * 0.02).astype(np.float16)
        wd = (rng.randn(inter, dim) * 0.02).astype(np.float16)
        lnw = np.ones(dim, dtype=np.float32)
        lnb = np.zeros(dim, dtype=np.float32)
        _backend.load_layer(i, wq, wp, wf, wd, lnw, lnb, lnw, lnb,
                            None, None, None, None)

def npu_vit_forward(x):
    """Run ViT encoder on NPU. x: jnp.ndarray [seq, dim], returns jnp.ndarray."""
    if _backend is None:
        raise RuntimeError("Call load_vit_weights() or load_random_weights() first")
    x_np = np.ascontiguousarray(np.asarray(x), dtype=np.float32)
    out = _backend.forward(x_np)
    return jnp.array(out)

# ---- LM generation (GPT-2 fast path) ----

def npu_lm_generate(prompt_ids, n_tokens=20, temperature=0.8, top_k=40):
    """Generate text on the NPU using the C KV-cached forward path.

    prompt_ids: list of int (token IDs) or jnp.ndarray
    n_tokens: number of tokens to generate
    Returns: list of generated token IDs (excluding prompt)

    Uses the same C fast path as NPULM (28ms/token, 36 tok/s).
    Must call load_lm_weights() first.
    """
    from opennpu.lm import NPUModel
    if not hasattr(_backend, '_lm'):
        raise RuntimeError("Call load_lm_weights() first")
    lm = _backend._lm
    if hasattr(prompt_ids, '__iter__') and not isinstance(prompt_ids, (list, tuple)):
        prompt_ids = list(np.asarray(prompt_ids))
    return lm.generate(list(prompt_ids), max_new_tokens=n_tokens,
                       temperature=temperature, top_k=top_k)

def load_lm_weights(model_name="gpt2", max_seq=1024):
    """Load GPT-2 weights onto NPU for fast generation.

    Downloads from HuggingFace, extracts weights, loads onto NPU CNA cache.
    Call once before npu_lm_generate().
    """
    from opennpu.lm import NPUModel, extract_gpt2_weights
    import tempfile, os
    global _backend
    _backend = _NPUBackend.get()
    lm = NPUModel(model_name=model_name, max_seq=max_seq)
    w_f32, w_f16 = extract_gpt2_weights(model_name)
    params_path = os.path.join(tempfile.gettempdir(), f"{model_name}_w_params.bin")
    lm.load_weights(w_f32, w_f16, params_path)
    _backend._lm = lm


def load_llama_weights(model):
    """Load a HuggingFace LLaMA/Qwen model onto the NPU for fast generation.

    model: transformers LLaMAForCausalLM (or similar). Must have
    q_proj/k_proj/v_proj/o_proj + mlp.gate_proj/up_proj/down_proj.
    Call once before npu_lm_generate().
    """
    from opennpu.lm import load_llama_weights as _load_llama
    global _backend
    _backend = _NPUBackend.get()
    lm = _load_llama(model)
    _backend._lm = lm
