import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opennpu.lm import NPUModel, extract_gpt2_weights

def main():
    from transformers import GPT2Tokenizer
    model_name = "gpt2"
    tok = GPT2Tokenizer.from_pretrained(model_name)
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The first time I saw the new iPhone 6S, I was a little disappointed."
    n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    print("Extracting weights...")
    W, W16 = extract_gpt2_weights(model_name)
    model = NPUModel(model_name, plugin_lib=os.environ.get("NPU_PLUGIN_LIB"))
    model.load_weights(W, W16, "/tmp/gpt2_params.bin")
    ids = tok(prompt)["input_ids"]
    print(f"prompt: {prompt[:60]!r}...  prefill {len(ids)} tokens, gen {n_gen}")

    t0 = time.time()
    gen = model.generate(ids, max_new_tokens=n_gen, temperature=0.0)
    dt = time.time() - t0
    print("NPU gen:", repr(tok.decode(gen)))
    print(f"decode: {dt:.2f}s  ({dt/n_gen*1000:.1f}ms/token, {n_gen/dt:.1f} tok/s)")
    prof = model.profile()
    print(f"[prof] mm={prof['mm_ms']:.1f} ln={prof['ln_ms']:.1f} attn={prof['attn_ms']:.1f} lh={prof['lh_ms']:.1f} step={prof['step_ms']:.1f} ms/step")
    model.close()
main()