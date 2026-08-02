import torch, numpy as np, sys
from transformers import GPT2Model
base = GPT2Model.from_pretrained("gpt2", torch_dtype=torch.float32)
sd = base.state_dict()
H = 768; NL = 12; INTER = 3072; V = 50257; MAXSEQ = 1024
def t(k): return sd[k].numpy().astype(np.float32)
W = {}
W["wte"] = t("wte.weight"); W["wpe"] = t("wpe.weight")[:MAXSEQ]
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
out = sys.argv[1] if len(sys.argv) > 1 else "gpt2_w_hybrid.npz"
np.savez(out, **W)
W16 = {}
for i in range(NL):
    W16[f"qkv{i}"] = W[f"qkv{i}"].astype(np.float16)
    W16[f"o{i}"] = W[f"o{i}"].astype(np.float16)
    W16[f"fc{i}"] = W[f"fc{i}"].astype(np.float16)
    for b in range(4): W16[f"pj{i}_b{b}"] = W[f"pj{i}_b{b}"].astype(np.float16)
out16 = sys.argv[2] if len(sys.argv) > 2 else "gpt2_w16_hybrid.npz"
np.savez(out16, **W16)
print(f"saved {out}: {len(W)} arrays, {out16}: {len(W16)} fp16 weights")
