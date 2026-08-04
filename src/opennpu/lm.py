import ctypes, os, time, numpy as np

_LIB = None
def _get_lib():
    global _LIB
    if _LIB is None:
        path = os.environ.get("NPU_PLUGIN_LIB", "libpjrt_npu.so")
        _LIB = ctypes.CDLL(path)
        _setup_signatures(_LIB)
    return _LIB

def _setup_signatures(lib):
    for name, restype, argtypes in [
        ("npu_cna_cache_setup", ctypes.c_int, [ctypes.c_int]*4),
        ("npu_cna_cache_load", ctypes.c_int, [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
        ("npu_cna_close", None, []),
        ("npu_lm_init", ctypes.c_int, [ctypes.c_int]*6 + [ctypes.c_float]),
        ("npu_lm_init2", ctypes.c_int, [ctypes.c_int]*6 + [ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]),
        ("npu_lm_load_slot", ctypes.c_int, [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
        ("npu_lm_load_norm", ctypes.c_int, [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]),
        ("npu_lm_load_embed", ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]),
        ("npu_lm_load_final_norm", ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p]),
        ("npu_lm_load_params", ctypes.c_int, [ctypes.c_char_p]),
        ("npu_lm_reset", None, []),
        ("npu_lm_step", ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]),
        ("npu_lm_prefill", ctypes.c_int, [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]),
        ("npu_lm_profile", None, [ctypes.POINTER(ctypes.c_double*6)]),
        ("npu_lm_close", None, []),
    ]:
        fn = getattr(lib, name); fn.restype = restype; fn.argtypes = argtypes

class NPUModel:
    _ARCH = {
        "gpt2": {"H":768, "NH":12, "DH":64, "INTER":3072, "V":50257, "NL":12, "MAXSEQ":1024},
    }

    _ARCH = {
        "gpt2": {"H":768, "NH":12, "DH":64, "INTER":3072, "V":50257, "NL":12,
                 "MAXSEQ":1024, "rope":False, "swiglu":False, "rmsnorm":False},
        "llama": {"H":4096, "NH":32, "DH":128, "INTER":11008, "V":32000, "NL":32,
                  "MAXSEQ":2048, "rope":True, "swiglu":True, "rmsnorm":True},
    }

    def __init__(self, model_name="gpt2", max_seq=None, plugin_lib=None,
                 dim=None, n_heads=None, inter=None, vocab=None, n_layers=None,
                 use_rope=None, use_swiglu=None, use_rmsnorm=None):
        if plugin_lib:
            os.environ["NPU_PLUGIN_LIB"] = plugin_lib
        if model_name in self._ARCH:
            self.cfg = dict(self._ARCH[model_name])
        else:
            # Custom architecture specified via kwargs
            self.cfg = {"H": dim or 768, "NH": n_heads or 12, "DH": (dim or 768)//(n_heads or 12),
                        "INTER": inter or 3072, "V": vocab or 32000, "NL": n_layers or 12,
                        "MAXSEQ": max_seq or 2048, "rope": use_rope or False,
                        "swiglu": use_swiglu or False, "rmsnorm": use_rmsnorm or False}
        if max_seq:
            self.cfg["MAXSEQ"] = max_seq
        self.lib = _get_lib()
        self._loaded = False
        self._nw = 0

    def load_weights(self, weights_f32, weights_f16, params_path):
        NL = self.cfg["NL"]; H = self.cfg["H"]
        Wnames, Wshapes = [], []
        for i in range(NL):
            Wnames += [f"qkv{i}", f"o{i}", f"fc{i}"] + [f"pj{i}_b{b}" for b in range(4)]
            Wshapes += [(H, self.cfg["H"]*3), (H, H), (H, self.cfg["INTER"])] + [(H, H)]*4
        self._nw = len(Wnames)
        self.lib.npu_lm_init(self.cfg["H"], self.cfg["NH"], self.cfg["INTER"],
                             self.cfg["V"], self.cfg["NL"], self.cfg["MAXSEQ"], 1e-5)
        for idx_w, nm in enumerate(Wnames):
            a = np.ascontiguousarray(weights_f16[nm].astype(np.float16))
            K, N = Wshapes[idx_w]
            if self.lib.npu_cna_cache_load(idx_w, a.ctypes.data_as(ctypes.c_void_p), K, N):
                raise RuntimeError(f"Failed to load weight {nm}")
        parts = self._build_params(weights_f32)
        buf = b"".join(p.tobytes() for p in parts)
        with open(params_path, "wb") as f: f.write(buf)
        if self.lib.npu_lm_load_params(params_path.encode()):
            raise RuntimeError("Failed to load params")
        self._loaded = True

    def _build_params(self, W):
        NL = self.cfg["NL"]; H = self.cfg["H"]; MAXSEQ = self.cfg["MAXSEQ"]
        parts = [W["wte"].astype(np.float32).ravel(), W["wpe"][:MAXSEQ].astype(np.float32).ravel()]
        for i in range(NL):
            for k in ["ln1w","ln1b","ln2w","ln2b"]:
                parts.append(np.ascontiguousarray(W[f"{k}{i}"]).astype(np.float32).ravel())
            parts.append(np.ascontiguousarray(W[f"qkvb{i}"]).astype(np.float32).ravel())
            parts.append(np.ascontiguousarray(W[f"ob{i}"]).astype(np.float32).ravel())
            parts.append(np.ascontiguousarray(W[f"fcb{i}"]).astype(np.float32).ravel())
            parts.append(np.ascontiguousarray(W[f"pjb{i}"]).astype(np.float32).ravel())
        parts.append(np.ascontiguousarray(W["lnf_w"]).astype(np.float32).ravel())
        parts.append(np.ascontiguousarray(W["lnf_b"]).astype(np.float32).ravel())
        return parts

    def reset(self):
        self.lib.npu_lm_reset()

    def step(self, token_id):
        logits = (ctypes.c_float * self.cfg["V"])()
        rc = self.lib.npu_lm_step(int(token_id), logits)
        if rc: raise RuntimeError(f"step failed: {rc}")
        return np.frombuffer(logits, dtype=np.float32).copy()

    def prefill(self, token_ids):
        logits = (ctypes.c_float * self.cfg["V"])()
        arr = (ctypes.c_int * len(token_ids))(*token_ids)
        rc = self.lib.npu_lm_prefill(arr, len(token_ids), logits)
        if rc: raise RuntimeError(f"prefill failed: {rc}")
        return np.frombuffer(logits, dtype=np.float32).copy()

    def generate(self, token_ids, max_new_tokens=50, temperature=0.0, top_k=0, top_p=1.0):
        self.reset()
        logits = self.prefill(token_ids)
        gen = []
        for _ in range(max_new_tokens):
            if temperature == 0.0:
                next_id = int(logits.argmax())
            else:
                logits_t = logits / temperature
                if top_k > 0:
                    idx = np.argpartition(logits_t, -top_k)[-top_k:]
                    mask = np.full_like(logits_t, -np.inf); mask[idx] = logits_t[idx]; logits_t = mask
                if top_p < 1.0:
                    sorted_idx = np.argsort(logits_t)[::-1]
                    cumprob = np.cumsum(np.exp(logits_t[sorted_idx] - logits_t.max()))
                    cutoff = np.searchsorted(cumprob, top_p)
                    keep = sorted_idx[:cutoff+1]
                    mask = np.full_like(logits_t, -np.inf); mask[keep] = logits_t[keep]; logits_t = mask
                probs = np.exp(logits_t - logits_t.max()); probs /= probs.sum()
                next_id = int(np.random.choice(len(probs), p=probs))
            gen.append(next_id)
            logits = self.step(next_id)
        return gen

    def profile(self):
        prof = (ctypes.c_double * 6)(); self.lib.npu_lm_profile(prof)
        return {"mm_ms": prof[0], "ln_ms": prof[1], "attn_ms": prof[2], "lh_ms": prof[4], "step_ms": prof[5]}

    def close(self):
        self.lib.npu_lm_close()


def extract_gpt2_weights(model_name="gpt2", out_f32="/tmp/gpt2_w_hybrid.npz", out_f16="/tmp/gpt2_w16_hybrid.npz"):
    import torch
    from transformers import GPT2Model
    base = GPT2Model.from_pretrained(model_name, torch_dtype=torch.float32)
    sd = base.state_dict()
    H, NL, INTER, V = 768, 12, 3072, 50257
    def t(k): return sd[k].numpy().astype(np.float32)
    W = {}
    W["wte"] = t("wte.weight"); W["wpe"] = t("wpe.weight")[:1024]
    W["lnf_w"] = t("ln_f.weight"); W["lnf_b"] = t("ln_f.bias")
    for i in range(NL):
        p = f"h.{i}."
        W[f"qkv{i}"] = t(p+"attn.c_attn.weight")
        W[f"qkvb{i}"] = t(p+"attn.c_attn.bias")
        W[f"o{i}"] = t(p+"attn.c_proj.weight"); W[f"ob{i}"] = t(p+"attn.c_proj.bias")
        W[f"ln1w{i}"] = t(p+"ln_1.weight"); W[f"ln1b{i}"] = t(p+"ln_1.bias")
        W[f"ln2w{i}"] = t(p+"ln_2.weight"); W[f"ln2b{i}"] = t(p+"ln_2.bias")
        W[f"fc{i}"] = t(p+"mlp.c_fc.weight"); W[f"fcb{i}"] = t(p+"mlp.c_fc.bias")
        pj = t(p+"mlp.c_proj.weight"); W[f"pjb{i}"] = t(p+"mlp.c_proj.bias")
        for b in range(4): W[f"pj{i}_b{b}"] = pj[b*H:(b+1)*H, :]
    np.savez(out_f32, **W)
    W16 = {}
    for i in range(NL):
        W16[f"qkv{i}"] = W[f"qkv{i}"].astype(np.float16)
        W16[f"o{i}"] = W[f"o{i}"].astype(np.float16)
        W16[f"fc{i}"] = W[f"fc{i}"].astype(np.float16)
        for b in range(4): W16[f"pj{i}_b{b}"] = W[f"pj{i}_b{b}"].astype(np.float16)
    np.savez(out_f16, **W16)
    return W, W16


def load_and_generate(prompt, model_name="gpt2", max_tokens=50, temperature=0.7, top_k=40, plugin_lib=None):
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained(model_name)
    W, W16 = extract_gpt2_weights(model_name)
    model = NPUModel(model_name, plugin_lib=plugin_lib)
    model.load_weights(W, W16, "/tmp/gpt2_params.bin")
    ids = tok(prompt)["input_ids"]
    gen = model.generate(ids, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    model.close()
    return tok.decode(gen)

def _load_llama_weights_from_hf(model, npu_model):
    """Load a HuggingFace LLaMA/Qwen-style model onto the NPU.

    Extracts weights directly from the torch model and loads them
    into the CNA cache using the generic per-slot loader.
    """
    import torch, numpy as np
    cfg = model.config
    D = getattr(cfg, 'hidden_size', 4096)
    NH = getattr(cfg, 'num_attention_heads', 32)
    INTER = getattr(cfg, 'intermediate_size', 11008)
    V = getattr(cfg, 'vocab_size', 32000)
    NL = getattr(cfg, 'num_hidden_layers', 32)
    HD = D // NH

    sd = model.state_dict()
    def t(k): return sd[k].detach().cpu().numpy().astype(np.float32)

    npu_model.cfg["H"] = D; npu_model.cfg["NH"] = NH; npu_model.cfg["DH"] = HD
    npu_model.cfg["INTER"] = INTER; npu_model.cfg["V"] = V; npu_model.cfg["NL"] = NL
    npu_model.lib.npu_lm_init2(D, NH, INTER, V, NL, npu_model.cfg["MAXSEQ"], 1e-5, 1, 1, 1, 10000.0)

    # Embeddings + final norm
    lm = npu_model.lib
    wte = np.ascontiguousarray(t("model.embed_tokens.weight"))
    lm.npu_lm_load_embed(wte.ctypes.data_as(ctypes.c_void_p), None, V, 0)
    lnf = np.ascontiguousarray(t("model.norm.weight"))
    lm.npu_lm_load_final_norm(lnf.ctypes.data_as(ctypes.c_void_p), None)

    # Per-layer
    for i in range(NL):
        p = f"model.layers.{i}."
        wq = np.ascontiguousarray(t(p+"self_attn.q_proj.weight").T, dtype=np.float16)
        wk = np.ascontiguousarray(t(p+"self_attn.k_proj.weight").T, dtype=np.float16)
        wv = np.ascontiguousarray(t(p+"self_attn.v_proj.weight").T, dtype=np.float16)
        wo = np.ascontiguousarray(t(p+"self_attn.o_proj.weight").T, dtype=np.float16)
        wg = np.ascontiguousarray(t(p+"mlp.gate_proj.weight").T, dtype=np.float16)
        wu = np.ascontiguousarray(t(p+"mlp.up_proj.weight").T, dtype=np.float16)
        wd = np.ascontiguousarray(t(p+"mlp.down_proj.weight").T, dtype=np.float16)
        n1 = np.ascontiguousarray(t(p+"input_layernorm.weight"), dtype=np.float32)
        n2 = np.ascontiguousarray(t(p+"post_attention_layernorm.weight"), dtype=np.float32)
        lm.npu_lm_load_slot(i, 0, wq.ctypes.data_as(ctypes.c_void_p), D, D)
        lm.npu_lm_load_slot(i, 1, wk.ctypes.data_as(ctypes.c_void_p), D, D)
        lm.npu_lm_load_slot(i, 2, wv.ctypes.data_as(ctypes.c_void_p), D, D)
        lm.npu_lm_load_slot(i, 3, wo.ctypes.data_as(ctypes.c_void_p), D, D)
        lm.npu_lm_load_slot(i, 4, wg.ctypes.data_as(ctypes.c_void_p), D, INTER)
        lm.npu_lm_load_slot(i, 5, wu.ctypes.data_as(ctypes.c_void_p), D, INTER)
        lm.npu_lm_load_slot(i, 6, wd.ctypes.data_as(ctypes.c_void_p), INTER, D)
        lm.npu_lm_load_norm(i, n1.ctypes.data_as(ctypes.c_void_p), None,
                            n2.ctypes.data_as(ctypes.c_void_p), None)
        # Keep arrays alive
        if not hasattr(npu_model, "_refs"): npu_model._refs = []
        npu_model._refs.extend([wq,wk,wv,wo,wg,wu,wd,n1,n2,wte,lnf])

    npu_model._loaded = True
    npu_model._arch = "llama"


def load_llama_weights(model):
    """Convenience: load a HuggingFace LLaMA model onto the NPU.

    Returns an NPUModel ready for npu_lm_generate.
    """
    npu = NPUModel(model_name="llama")
    _load_llama_weights_from_hf(model, npu)
    return npu
