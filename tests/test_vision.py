"""Smoke test: NPU plugin loads and vision encoder runs."""
import os, sys, ctypes, numpy as np

def test_plugin_loads():
    path = os.environ.get("NPU_PLUGIN_LIB",
        os.path.join(os.path.dirname(__file__), "..", "src", "opennpu", "pjrt_c", "libpjrt_npu.so"))
    p = ctypes.CDLL(path)
    p.npu_cna_cache_setup.restype = ctypes.c_int
    p.npu_cna_cache_setup.argtypes = [ctypes.c_int] * 4
    assert p.npu_cna_cache_setup(4, 64, 768, 768) == 0
    p.npu_cna_close.restype = None
    p.npu_cna_close()
    print("PASS: plugin loads and CNA initializes")

def test_vision_encoder():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from opennpu.torch_npu import NPUViTEncoder
    enc = NPUViTEncoder.random(n_layers=12, seq_len=197)
    x = np.random.randn(197, 768).astype(np.float32) * 0.1
    out = enc._backend.forward(x)
    assert out.shape == (197, 768), f"Expected (197, 768), got {out.shape}"
    assert np.isfinite(out).all(), "Output contains NaN/Inf"
    print(f"PASS: vision encoder produces {out.shape} output, mean={out.mean():.4f}")

if __name__ == "__main__":
    test_plugin_loads()
    test_vision_encoder()
    print("\nAll tests passed.")
