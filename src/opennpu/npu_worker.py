#!/usr/bin/env python3
"""Per-op NPU subprocess worker for the ONNX / PyTorch runners.

Invoked as:  python -m opennpu.npu_worker <job.pkl> <out.pkl>

job = {
    "op":        "matmul" | "add" | "relu" | ...   (opennpu op name)
    "precision": "float16" | "int8" | "int16",
    "inputs":    [np.ndarray(fp32), ...],          # host tensors
}
out = np.ndarray(fp32)                              # single output tensor

The NPU device is exclusive (one fd per process) and a single fd's IOVA pool
holds only one large op in the high-DMA window, so each NPU op runs in its OWN
fresh process. Tensors flow through pickle files. No librknnrt at runtime.
"""
import os
import sys
import pickle

import numpy as np

sys.path.insert(0, "/home/poad42/opennpu/src")
from opennpu.pjrt import PJRTClient

DUMPS = "/home/poad42/dumps_{op}"
TPL = "/home/poad42/templates"

CODEGEN_OPS = {"add", "sub", "mul", "div"}
# Elementwise activations tile to any shape via the captured (1,64,768) kernel.
TILEABLE_ACT = {"relu", "sigmoid", "tanh", "leakyrelu", "gelu"}
SUFFIX = {"float16": "", "int8": "_w8a8", "int16": "_w16a16i"}

# Matmul is shape-pinned (M,K_inner,N baked in the regcmd). Each distinct (X, W)
# shape needs its own captured template. Map (Xshape, Wshape) -> template name.
MATMUL_BY_SHAPE = {
    ((1, 64, 768),  (768, 3072)): "matmul",        # FFN-up
    ((1, 64, 768),  (768, 768)):  "mm_768x768",   # QKV / O-proj
    ((1, 64, 3072), (3072, 768)): "mm_3072x768",  # FFN-down
    ((1, 64, 768),  (768, 64)):   "mm_768x64",    # attn QK^T
    ((1, 64, 64),   (64, 768)):   "mm_64x768",    # attn @V
}


def _template_dir(op, precision):
    sfx = SUFFIX[precision]
    name = op + sfx
    d = os.path.join(TPL, name)
    if os.path.isdir(d):
        return d
    return os.path.join(TPL, op)


def run(job):
    op = job["op"]
    prec = job.get("precision", "float16")
    inputs = [np.ascontiguousarray(t, dtype=np.float32) for t in job["inputs"]]
    client = PJRTClient()
    try:
        if op in CODEGEN_OPS and prec == "float16":
            exe = client.compile_codegen(op, shape=inputs[0].shape,
                                         dumps_dir=DUMPS.format(op=op))
            out_shape = inputs[0].shape
        elif op == "matmul":
            key = (tuple(inputs[0].shape), tuple(inputs[1].shape))
            tname = MATMUL_BY_SHAPE.get(key)
            if tname is None:
                raise RuntimeError(f"matmul: no captured template for X={inputs[0].shape} W={inputs[1].shape}")
            tdir = os.path.join(TPL, tname)
            with open(os.path.join(tdir, "meta.json")) as f:
                import json
                meta = json.load(f)
            cap_shape = tuple(meta["shape"])
            exe = client.compile_template(op, tdir, shape=cap_shape)
            out_shape = cap_shape
        else:
            tdir = _template_dir(op, prec)
            with open(os.path.join(tdir, "meta.json")) as f:
                import json
                meta = json.load(f)
            cap_shape = tuple(meta["shape"])
            exe = client.compile_template(op, tdir, shape=cap_shape)
            out_shape = inputs[0].shape if op in TILEABLE_ACT else cap_shape
        pjrt_in = [client.buffer_from_host(t.astype(np.float16)) for t in inputs]
        out_buf = exe.execute(*pjrt_in)
        out = np.ascontiguousarray(out_buf.to_host().astype(np.float32))
        if out.shape != tuple(out_shape):
            out = out.reshape(out_shape)
        return out
    finally:
        client.destroy()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        serve(); sys.exit(0)
    job_path, out_path = sys.argv[1], sys.argv[2]
    with open(job_path, "rb") as f:
        job = pickle.load(f)
    result = run(job)
    with open(out_path, "wb") as f:
        pickle.dump(result, f)


def serve():
    """Persistent server mode: read length-prefixed pickled jobs from stdin,
    write length-prefixed pickled outputs to stdout. One process for many ops;
    each op opens a fresh PJRTClient (close+reopen resets the IOVA pool to the
    top, so every op gets the high-DMA window) -- avoids the ~0.3s python
    startup per op that dominates the subprocess-per-op runner."""
    import struct
    sys.stdout.reconfigure(newline=b"")
    sys.stdin.reconfigure(newline=b"")
    r = sys.stdin.buffer
    w = sys.stdout.buffer
    while True:
        hdr = r.read(4)
        if len(hdr) < 4:
            break
        n = struct.unpack("<I", hdr)[0]
        job = pickle.loads(r.read(n))
        try:
            result = run(job)
        except Exception as e:
            result = {"__error__": str(e)}
        blob = pickle.dumps(result)
        w.write(struct.pack("<I", len(blob)) + blob); w.flush()