"""PyTorch nn.Module integration for OpenNPU.

Provides drop-in torch.nn.Module replacements that run on the NPU.
People use standard PyTorch API — no ctypes, no custom calls.

Usage:
    from opennpu.torch_npu import NPUViTEncoder
    encoder = NPUViTEncoder.from_pretrained("google/vit-base-patch16-224")
    out = encoder(x)  # standard forward, runs on NPU

    # Or replace blocks in an existing model:
    model = ViTModel.from_pretrained(...)
    model.encoder = NPUViTEncoder.from_torch(model)
    out = model(x)
"""
import os
import ctypes
import numpy as np
import torch
import torch.nn as nn

_PLUGIN_PATH = os.environ.get("NPU_PLUGIN_LIB",
    os.path.join(os.path.dirname(__file__), "pjrt_c", "libpjrt_npu.so"))


class _NPUBackend:
    """Singleton wrapper around the C vision_forward backend."""
    _instance = None

    def __init__(self):
        self.plugin = ctypes.CDLL(_PLUGIN_PATH)
        p = self.plugin
        p.npu_vision_init.restype = ctypes.c_int
        p.npu_vision_init.argtypes = [ctypes.c_int] * 5 + [ctypes.c_float]
        p.npu_vision_load_weights.restype = ctypes.c_int
        p.npu_vision_load_weights.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 12
        p.npu_vision_forward.restype = ctypes.c_int
        p.npu_vision_forward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        p.npu_vision_profile.restype = None
        p.npu_vision_profile.argtypes = [ctypes.POINTER(ctypes.c_double)] * 6
        p.npu_vision_reset_profile.restype = None
        p.npu_vision_reset_profile.argtypes = []
        p.npu_vision_close.restype = None
        p.npu_cna_cache_setup.restype = ctypes.c_int
        p.npu_cna_cache_setup.argtypes = [ctypes.c_int] * 4
        p.npu_cna_cache_load.restype = ctypes.c_int
        p.npu_cna_cache_load.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        p.npu_cna_cache_run_m.restype = ctypes.c_int
        p.npu_cna_cache_run_m.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        p.npu_cna_wt_n.restype = ctypes.c_int
        p.npu_cna_wt_n.argtypes = [ctypes.c_int]
        p.npu_cna_close.restype = None
        self._initialized = False

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def init_vision(self, dim=768, n_heads=12, inter=3072, max_seq=197, max_layers=12, eps=1e-6):
        if not self._initialized:
            rc = self.plugin.npu_vision_init(dim, n_heads, inter, max_seq, max_layers, eps)
            if rc != 0:
                raise RuntimeError(f"NPU vision init failed: {rc}")
            self._initialized = True
            self._dim = dim; self._inter = inter; self._n_heads = n_heads; self.n_layers = max_layers

    def load_layer(self, i, wq, wp, wf, wd, ln1w, ln1b, ln2w, ln2b,
                   qkv_b=None, proj_b=None, fc_b=None, down_b=None):
        def p(x, dt): return np.ascontiguousarray(x, dtype=dt)
        ln1w_a = p(ln1w, np.float32); ln1b_a = p(ln1b, np.float32)
        ln2w_a = p(ln2w, np.float32); ln2b_a = p(ln2b, np.float32)
        qkv_b_a = p(qkv_b, np.float32) if qkv_b is not None else None
        proj_b_a = p(proj_b, np.float32) if proj_b is not None else None
        fc_b_a = p(fc_b, np.float32) if fc_b is not None else None
        down_b_a = p(down_b, np.float32) if down_b is not None else None
        if not hasattr(self, '_ln_refs'):
            self._ln_refs = []
        refs = [ln1w_a, ln1b_a, ln2w_a, ln2b_a]
        if qkv_b_a is not None: refs.append(qkv_b_a)
        if proj_b_a is not None: refs.append(proj_b_a)
        if fc_b_a is not None: refs.append(fc_b_a)
        if down_b_a is not None: refs.append(down_b_a)
        self._ln_refs.extend(refs)
        self.plugin.npu_vision_load_weights(
            i,
            p(wq, np.float16).ctypes.data_as(ctypes.c_void_p),
            p(wp, np.float16).ctypes.data_as(ctypes.c_void_p),
            p(wf, np.float16).ctypes.data_as(ctypes.c_void_p),
            p(wd, np.float16).ctypes.data_as(ctypes.c_void_p),
            ln1w_a.ctypes.data_as(ctypes.c_void_p),
            ln1b_a.ctypes.data_as(ctypes.c_void_p),
            ln2w_a.ctypes.data_as(ctypes.c_void_p),
            ln2b_a.ctypes.data_as(ctypes.c_void_p),
            qkv_b_a.ctypes.data_as(ctypes.c_void_p) if qkv_b_a is not None else None,
            proj_b_a.ctypes.data_as(ctypes.c_void_p) if proj_b_a is not None else None,
            fc_b_a.ctypes.data_as(ctypes.c_void_p) if fc_b_a is not None else None,
            down_b_a.ctypes.data_as(ctypes.c_void_p) if down_b_a is not None else None,
        )

    def forward(self, x_np):
        x_np = np.ascontiguousarray(x_np, dtype=np.float32)
        seq_len = x_np.shape[0]
        out = np.zeros(x_np.shape, dtype=np.float32)
        rc = self.plugin.npu_vision_forward(
            x_np.ctypes.data_as(ctypes.c_void_p),
            seq_len, self.n_layers,
            out.ctypes.data_as(ctypes.c_void_p))
        if rc:
            raise RuntimeError(f"NPU forward failed: {rc}")
        return out

    def profile(self):
        t = [ctypes.c_double() for _ in range(6)]
        self.plugin.npu_vision_profile(*[ctypes.byref(x) for x in t])
        return {k: v.value for k, v in zip(
            ['npu_mm', 'ln', 'attn', 'gelu', 'res', 'total'], t)}

    def reset_profile(self):
        self.plugin.npu_vision_reset_profile()


class NPUViTEncoder(nn.Module):
    """Drop-in replacement for a HuggingFace ViT encoder.

    Standard PyTorch nn.Module. Accepts torch.Tensor, returns torch.Tensor.
    All 48 matmuls run on NPU; attention/GELU/LN run on CPU with OpenMP.

    Usage:
        encoder = NPUViTEncoder.from_pretrained("google/vit-base-patch16-224")
        out = encoder(x)  # [1, 197, 768] -> [1, 197, 768]

        # Or from an existing model:
        model = ViTModel.from_pretrained(...)
        encoder = NPUViTEncoder.from_torch(model)
        out = encoder(x)
    """

    def __init__(self, n_layers=12, seq_len=197, dim=768, inter=3072, n_heads=12, eps=1e-6):
        super().__init__()
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.dim = dim
        self.inter = inter
        self.n_heads = n_heads
        self._backend = _NPUBackend.get()
        self._backend.init_vision(dim, n_heads, inter, max(seq_len, 256), n_layers, eps)
        self._weights_loaded = False

    @classmethod
    def from_torch(cls, model, **kwargs):
        """Create from a HuggingFace ViTModel, SiglipModel, or similar.

        Automatically reads model config (hidden_size, num_heads,
        intermediate_size, image_size, patch_size) to initialize
        the NPU vision encoder with the right dimensions.
        """
        if hasattr(model, 'vision_model'):
            vm = model.vision_model
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            vm = model
        elif hasattr(model, 'blocks'):
            vm = model
        else:
            raise ValueError("Expected SiglipModel, ViTModel, or model with .blocks")

        cfg = getattr(vm, 'config', None)
        dim = getattr(cfg, 'hidden_size', kwargs.get('dim', 768))
        n_heads = getattr(cfg, 'num_attention_heads', kwargs.get('n_heads', 12))
        inter = getattr(cfg, 'intermediate_size', kwargs.get('inter', 3072))
        if cfg and hasattr(cfg, 'image_size') and hasattr(cfg, 'patch_size'):
            seq_len = (cfg.image_size // cfg.patch_size) ** 2
        else:
            seq_len = kwargs.get('seq_len', 197)

        if hasattr(vm, 'encoder') and hasattr(vm.encoder, 'layers'):
            layers = list(vm.encoder.layers)
        elif hasattr(vm, 'encoder') and hasattr(vm.encoder, 'layer'):
            layers = list(vm.encoder.layer)
        elif hasattr(vm, 'blocks'):
            layers = list(vm.blocks)
        else:
            raise ValueError("Cannot find encoder layers")

        n_layers = kwargs.get('n_layers', len(layers))
        inst = cls(n_layers=n_layers, seq_len=seq_len, dim=dim, inter=inter, n_heads=n_heads)
        for i, block in enumerate(layers[:n_layers]):
            inst._load_block_weights(i, block)
        inst._weights_loaded = True
        return inst

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        """Load from HuggingFace hub."""
        from transformers import ViTModel
        model = ViTModel.from_pretrained(model_name)
        return cls.from_torch(model, **kwargs)

    @classmethod
    def random(cls, seed=42, **kwargs):
        """Random weights for benchmarking."""
        inst = cls(**kwargs)
        rng = np.random.RandomState(seed)
        for i in range(inst.n_layers):
            wq = (rng.randn(inst.dim, inst.dim*3) * 0.02).astype(np.float16)
            wp = (rng.randn(inst.dim, inst.dim) * 0.02).astype(np.float16)
            wf = (rng.randn(inst.dim, inst.inter) * 0.02).astype(np.float16)
            wd = (rng.randn(inst.inter, inst.dim) * 0.02).astype(np.float16)
            lnw = np.ones(inst.dim, dtype=np.float32)
            lnb = np.zeros(inst.dim, dtype=np.float32)
            inst._backend.load_layer(i, wq, wp, wf, wd, lnw, lnb, lnw, lnb,
                                     None, None, None, None)
        inst._weights_loaded = True
        return inst

    def _load_block_weights(self, i, block):
        def w(t): return t.detach().cpu().numpy()

        # SigLIP: self_attn.{q,k,v,out}_proj, layer_norm{1,2}, mlp.{fc1,fc2}
        if hasattr(block, 'self_attn') and hasattr(block.self_attn, 'q_proj'):
            wq = w(block.self_attn.q_proj.weight).T
            wk = w(block.self_attn.k_proj.weight).T
            wv = w(block.self_attn.v_proj.weight).T
            w_qkv = np.concatenate([wq, wk, wv], axis=1)
            w_proj = w(block.self_attn.out_proj.weight).T
        # HuggingFace ViT: attention.attention.{query,key,value}, attention.output.dense
        elif hasattr(block, 'attention') and hasattr(block.attention, 'attention'):
            attn = block.attention.attention
            wq, wk, wv = w(attn.query.weight).T, w(attn.key.weight).T, w(attn.value.weight).T
            w_qkv = np.concatenate([wq, wk, wv], axis=1)
            w_proj = w(block.attention.output.dense.weight).T
        # timm ViT: attn.qkv, attn.proj
        elif hasattr(block, 'attn'):
            w_qkv = w(block.attn.qkv.weight).T
            w_proj = w(block.attn.proj.weight).T
        else:
            raise ValueError(f"Unknown attention structure at layer {i}")

        # MLP
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'fc1'):
            w_fc = w(block.mlp.fc1.weight).T
            w_down = w(block.mlp.fc2.weight).T
        elif hasattr(block, 'intermediate'):
            w_fc = w(block.intermediate.dense.weight).T
            w_down = w(block.output.dense.weight).T
        else:
            raise ValueError(f"Unknown MLP structure at layer {i}")

        # LayerNorm
        if hasattr(block, 'layer_norm1'):
            ln1w, ln1b = w(block.layer_norm1.weight), w(block.layer_norm1.bias)
            ln2w, ln2b = w(block.layer_norm2.weight), w(block.layer_norm2.bias)
        elif hasattr(block, 'layernorm_before'):
            ln1w, ln1b = w(block.layernorm_before.weight), w(block.layernorm_before.bias)
            ln2w, ln2b = w(block.layernorm_after.weight), w(block.layernorm_after.bias)
        elif hasattr(block, 'norm1'):
            ln1w, ln1b = w(block.norm1.weight), w(block.norm1.bias)
            ln2w, ln2b = w(block.norm2.weight), w(block.norm2.bias)
        else:
            raise ValueError(f"Unknown norm structure at layer {i}")

        # Extract biases
        if hasattr(block, 'self_attn') and hasattr(block.self_attn, 'q_proj'):
            qkv_b = w(block.self_attn.q_proj.bias)
            k_b = w(block.self_attn.k_proj.bias)
            v_b = w(block.self_attn.v_proj.bias)
            qkv_bias = np.concatenate([qkv_b, k_b, v_b])
            proj_b = w(block.self_attn.out_proj.bias)
            fc_b = w(block.mlp.fc1.bias)
            down_b = w(block.mlp.fc2.bias)
        elif hasattr(block, 'attn'):
            qkv_bias = w(block.attn.qkv.bias)
            proj_b = w(block.attn.proj.bias)
            fc_b = w(block.mlp.fc1.bias)
            down_b = w(block.mlp.fc2.bias)
        else:
            qkv_bias = proj_b = fc_b = down_b = None

        self._backend.load_layer(i, w_qkv, w_proj, w_fc, w_down,
                                 ln1w, ln1b, ln2w, ln2b,
                                 qkv_bias, proj_b, fc_b, down_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch forward pass. x: [batch, seq, dim] or [seq, dim]."""
        if not self._weights_loaded:
            raise RuntimeError("Weights not loaded. Use from_torch() or random().")

        x_np = np.ascontiguousarray(x.detach().cpu().numpy().astype(np.float32))
        squeeze = False
        if x_np.ndim == 3:
            x_np = np.ascontiguousarray(x_np[0])
            squeeze = True

        out = self._backend.forward(x_np)

        if squeeze:
            out = out[None, ...]
        return torch.from_numpy(out).to(x.device)

    def close(self):
        self._backend = None

    def benchmark(self, x=None, n_warmup=2, n_runs=5):
        """Returns timing dict."""
        import time
        if x is None:
            x = torch.randn(1, self.seq_len, self.dim) * 0.1
        for _ in range(n_warmup):
            self(x)
        self._backend.reset_profile()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            self(x)
        dt = (time.perf_counter() - t0) / n_runs * 1000
        prof = self._backend.profile()
        for k in prof:
            prof[k] = prof[k] / n_runs
        prof['total_ms'] = dt
        prof['throughput'] = 1000.0 / dt
        return prof

class NPULM:
    """PyTorch-compatible GPT-2 generation on the NPU.

    Uses the C KV-cached forward pass (28ms/token, 36 tok/s).
    All 84 matmuls per token run on the NPU via CNA descriptor.
    CPU handles attention softmax and lm_head (OpenMP, 3 A76 cores).

    Usage:
        from opennpu.torch_npu import NPULM
        lm = NPULM.from_pretrained("gpt2")
        tokens = lm.generate("The future of AI is", n_tokens=20)
        # tokens is a list of token IDs — decode with any tokenizer

        # Or from an existing model:
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        lm = NPULM.from_torch(model)
        tokens = lm.generate(prompt_ids, n_tokens=20)
    """

    def __init__(self, model_name="gpt2", max_seq=1024):
        from opennpu.lm import NPUModel
        self._npu = NPUModel(model_name=model_name, max_seq=max_seq)
        self._loaded = False
        self.model_name = model_name

    @classmethod
    def from_pretrained(cls, model_name="gpt2", max_seq=1024):
        """Load from HuggingFace hub. Extracts weights and loads onto NPU."""
        inst = cls(model_name=model_name, max_seq=max_seq)
        from opennpu.lm import extract_gpt2_weights
        import tempfile, os
        w_f32, w_f16 = extract_gpt2_weights(model_name)
        params_path = os.path.join(tempfile.gettempdir(), f"{model_name}_w_params.bin")
        inst._npu.load_weights(w_f32, w_f16, params_path)
        inst._loaded = True
        return inst

    @classmethod
    def from_torch(cls, model, max_seq=1024):
        """Load from a HuggingFace GPT2LMHeadModel."""
        model_name = getattr(model.config, '_name_or_path', 'gpt2')
        inst = cls(model_name=model_name, max_seq=max_seq)
        from opennpu.lm import extract_gpt2_weights
        import tempfile, os
        # Extract weights directly from the model object
        w_f32, w_f16 = _extract_gpt2_weights_from_model(model)
        params_path = os.path.join(tempfile.gettempdir(), f"{model_name}_w_params.bin")
        inst._npu.load_weights(w_f32, w_f16, params_path)
        inst._loaded = True
        return inst

    def generate(self, prompt, n_tokens=20, temperature=0.8, top_k=40):
        """Generate text. prompt can be a string (needs tokenizer) or list of token IDs.

        Returns list of generated token IDs (excluding prompt).
        """
        if not self._loaded:
            raise RuntimeError("Call from_pretrained() or from_torch() first")
        if isinstance(prompt, str):
            from transformers import GPT2Tokenizer
            tok = GPT2Tokenizer.from_pretrained(self.model_name)
            ids = tok.encode(prompt)
        else:
            ids = list(prompt)
        return self._npu.generate(ids, max_new_tokens=n_tokens,
                                  temperature=temperature, top_k=top_k)

    def step(self, token_id):
        """Single decode step. Returns logits as numpy array."""
        return self._npu.step(token_id)

    def prefill(self, token_ids):
        """Process prompt tokens. Returns logits for last token."""
        return self._npu.prefill(list(token_ids))

    def reset(self):
        self._npu.reset()

    def profile(self):
        return self._npu.profile()

    def close(self):
        self._npu.close()


def _extract_gpt2_weights_from_model(model):
    """Extract GPT-2 weights from a torch model (not from HuggingFace hub)."""
    import torch, numpy as np
    sd = model.state_dict()
    H, NL, INTER, V = 768, 12, 3072, 50257
    def t(k): return sd[k].numpy().astype(np.float32)
    W = {}
    W["wte"] = t("transformer.wte.weight"); W["wpe"] = t("transformer.wpe.weight")[:1024]
    W["lnf_w"] = t("transformer.ln_f.weight"); W["lnf_b"] = t("transformer.ln_f.bias")
    for i in range(NL):
        p = f"transformer.h.{i}."
        W[f"qkv{i}"] = t(p+"attn.c_attn.weight")
        W[f"qkvb{i}"] = t(p+"attn.c_attn.bias")
        W[f"o{i}"] = t(p+"attn.c_proj.weight"); W[f"ob{i}"] = t(p+"attn.c_proj.bias")
        W[f"ln1w{i}"] = t(p+"ln_1.weight"); W[f"ln1b{i}"] = t(p+"ln_1.bias")
        W[f"ln2w{i}"] = t(p+"ln_2.weight"); W[f"ln2b{i}"] = t(p+"ln_2.bias")
        W[f"fc{i}"] = t(p+"mlp.c_fc.weight"); W[f"fcb{i}"] = t(p+"mlp.c_fc.bias")
        pj = t(p+"mlp.c_proj.weight"); W[f"pjb{i}"] = t(p+"mlp.c_proj.bias")
        for b in range(4): W[f"pj{i}_b{b}"] = pj[b*H:(b+1)*H, :]
    W16 = {}
    for i in range(NL):
        W16[f"qkv{i}"] = W[f"qkv{i}"].astype(np.float16)
        W16[f"o{i}"] = W[f"o{i}"].astype(np.float16)
        W16[f"fc{i}"] = W[f"fc{i}"].astype(np.float16)
        for b in range(4): W16[f"pj{i}_b{b}"] = W[f"pj{i}_b{b}"].astype(np.float16)
    return W, W16
