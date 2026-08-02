#!/usr/bin/env python3
"""Open-source ONNX runtime backend for the RK3588 NPU.

Executes an ONNX model on the NPU using the opennpu raw-DRM-ioctl stack (no
librknnrt at runtime). Each NPU op runs in its own subprocess (the NPU device is
exclusive and one fd's IOVA pool holds only one large op in the high-DMA window);
unsupported ops / shape-mismatched ops fall back to a numpy CPU implementation so
mixed graphs still run.

The NPU templates are shape-specific (captured at fixed transformer-embedding
shapes): add/sub/mul/div [1,64,768], relu/sigmoid/tanh/leakyrelu/softmax
[1,64,768], matmul [1,64,768]@[768,3072]->[1,64,3072], concat 2x[1,32,768]->
[1,64,768] (axis=1), transpose [1,64,768]->[1,768,64] (perm=[0,2,1]).

Op Lowering Strategy (see docs/OP_LOWERING.md for full details):
  NPU-native:     Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, LeakyRelu, Clip,
                  Softmax, MatMul, Concat, Transpose, Gelu, LayerNorm(scale=1/B=0)
  NPU via CNA:    MatMul (arbitrary shapes via cna_matmul.c)
  CPU reduce+NPU: LayerNorm (trained scale/B -> CPU mean/var, CPU affine)
  CPU fallback:   Shape (compile-time), Slice, Gather, Erf(tanh-approx),
                  Reshape, Squeeze, Unsqueeze, Cast, Where, Equal, etc.

Usage (on the SBC):
    from opennpu.onnx_runner import ONNXRunner
    r = ONNXRunner("model.onnx")
    out = r.run({"input": x_fp32})      # {output_name: np.ndarray}

Or build a model in-process with the onnx API and call ONNXRunner(model_proto).
"""
import os
import sys
import pickle
import subprocess
import tempfile, shutil
from typing import Dict, Optional

import numpy as np

try:
    import onnx
    from onnx import numpy_helper, shape_inference
except Exception:
    onnx = None

sys.path.insert(0, "/home/poad42/opennpu/src")

WORKER = [sys.executable, "-m", "opennpu.npu_worker"]

# Fixed NPU template shapes (captured primitives).
# value: (npu_op, n_inputs, [input_shapes], output_shape)
NPU_OPS = {
    "Add":        ("add",       2, [(1, 64, 768), (1, 64, 768)], (1, 64, 768)),
    "Sub":        ("sub",       2, [(1, 64, 768), (1, 64, 768)], (1, 64, 768)),
    "Mul":        ("mul",       2, [(1, 64, 768), (1, 64, 768)], (1, 64, 768)),
    "Div":        ("div",       2, [(1, 64, 768), (1, 64, 768)], (1, 64, 768)),
    "Relu":       ("relu",      1, [(1, 64, 768)],               (1, 64, 768)),
    "Sigmoid":    ("sigmoid",   1, [(1, 64, 768)],               (1, 64, 768)),
    "Tanh":       ("tanh",      1, [(1, 64, 768)],               (1, 64, 768)),
    "LeakyRelu":  ("leakyrelu", 1, [(1, 64, 768)],               (1, 64, 768)),
    "Softmax":    ("softmax",   1, [(1, 64, 768)],               (1, 64, 768)),
    "MatMul":     ("matmul",    2, [(1, 64, 768), (768, 3072)],   (1, 64, 3072)),
    "Concat":     ("concat",    2, [(1, 32, 768), (1, 32, 768)],  (1, 64, 768)),
    "Transpose":  ("transpose", 1, [(1, 64, 768)],               (1, 768, 64)),
}


def _shape(t):
    return tuple(t.shape) if hasattr(t, "shape") else tuple()


# ── CPU fallback reference implementations (numpy) ───────────────
def _cpu_op(node_type, attrs, inputs):
    if node_type == "Add":
        return inputs[0] + inputs[1]
    if node_type == "Sub":
        return inputs[0] - inputs[1]
    if node_type == "Mul":
        return inputs[0] * inputs[1]
    if node_type == "Div":
        return inputs[0] / inputs[1]
    if node_type == "Relu":
        return np.maximum(inputs[0], 0)
    if node_type == "Sigmoid":
        return 1.0 / (1.0 + np.exp(-inputs[0]))
    if node_type == "Tanh":
        return np.tanh(inputs[0])
    if node_type == "LeakyRelu":
        a = attrs.get("alpha", 0.01)
        return np.where(inputs[0] > 0, inputs[0], a * inputs[0])
    if node_type == "Softmax":
        axis = attrs.get("axis", -1)
        x = inputs[0] - np.max(inputs[0], axis=axis, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=axis, keepdims=True)
    if node_type == "MatMul":
        return np.matmul(inputs[0], inputs[1])
    if node_type == "Concat":
        axis = attrs.get("axis", 1)
        return np.concatenate(inputs, axis=axis)
    if node_type == "Transpose":
        perm = attrs.get("perm", None)
        return np.transpose(inputs[0], perm)
    if node_type == "Gemm":
        a = attrs.get("alpha", 1.0); b = attrs.get("beta", 1.0)
        tA = attrs.get("transA", 0); tB = attrs.get("transB", 0)
        A = inputs[0].T if tA else inputs[0]
        B = inputs[1].T if tB else inputs[1]
        out = a * np.matmul(A, B)
        if len(inputs) > 2:
            out = out + b * inputs[2]
        return out
    if node_type == "Reshape":
        return inputs[0].reshape(inputs[1].astype(np.int64).tolist())
    if node_type == "ReduceMean":
        axes = attrs.get("axes", None)
        return np.mean(inputs[0], axis=tuple(axes) if axes else None,
                       keepdims=attrs.get("keepdims", 1) == 1)
    if node_type == "Sqrt":
        return np.sqrt(inputs[0])
    if node_type == "Exp":
        return np.exp(inputs[0])
    if node_type == "Pow":
        return np.power(inputs[0], inputs[1])
    if node_type == "Max":
        return np.maximum(inputs[0], inputs[1])
    if node_type == "Min":
        return np.minimum(inputs[0], inputs[1])
    if node_type == "Neg":
        return -inputs[0]
    if node_type == "Clip":
        lo = inputs[1] if len(inputs) > 1 else None
        hi = inputs[2] if len(inputs) > 2 else None
        return np.clip(inputs[0], lo, hi)
    if node_type == "Identity":
        return inputs[0]
    if node_type == "Constant":
        v = attrs.get("value")
        if v is not None:
            return np.ascontiguousarray(v, dtype=np.float32)
        return np.array(0.0, dtype=np.float32)
    if node_type == "Erf":
        x = inputs[0]
        return np.tanh(1.128379167 * x * (1.0 + 0.044715 * x ** 3))
    if node_type == "Gelu":
        return 0.5 * inputs[0] * (1.0 + np.tanh(
            0.7978845608 * (inputs[0] + 0.044715 * inputs[0] ** 3)))
    if node_type == "Squeeze":
        axes = attrs.get("axes", None)
        return np.squeeze(inputs[0], axis=tuple(axes) if axes else None)
    if node_type == "Unsqueeze":
        axes = attrs.get("axes", None) or inputs[1].astype(np.int64).tolist()
        x = inputs[0]
        for a in sorted(axes):
            x = np.expand_dims(x, axis=a)
        return x
    if node_type == "LayerNormalization":
        axis = attrs.get("axis", -1)
        eps = attrs.get("epsilon", 1e-5)
        x = inputs[0]
        mean = np.mean(x, axis=axis, keepdims=True)
        var = np.var(x, axis=axis, keepdims=True)
        out = (x - mean) / np.sqrt(var + eps)
        if len(inputs) > 1:
            out = out * inputs[1]
        if len(inputs) > 2:
            out = out + inputs[2]
        return out
    if node_type == "Gather":
        data, indices = inputs[0], inputs[1].astype(np.int64)
        return np.take(data, indices, axis=attrs.get("axis", 0))
    if node_type == "Cast":
        dt = attrs.get("to", 1)
        m = {1:np.float32,2:np.uint8,3:np.int8,6:np.int32,7:np.int64,9:np.bool_,
             10:np.float16,11:np.float64,12:np.int32,13:np.uint32}
        return inputs[0].astype(m.get(dt, np.float32))
    if node_type == "Equal":
        return (inputs[0] == inputs[1])
    if node_type == "Where":
        return np.where(inputs[0].astype(bool), inputs[1], inputs[2])
    if node_type == "Trilu":
        # upper/lower triangular ones; k = attrs.get('k',0); upper=attrs.get('upper',1)
        k = int(attrs.get("k", 0)) if attrs.get("k") is not None else 0
        upper = attrs.get("upper", 1)
        x = inputs[0]
        if upper:
            return np.triu(x, k=k)
        return np.tril(x, k=k)
    if node_type == "Range":
        start, limit, delta = inputs[0], inputs[1], inputs[2]
        return np.arange(start, limit, delta)
    if node_type == "Shape":
        return np.array(inputs[0].shape, dtype=np.int64)
    raise NotImplementedError(f"CPU fallback not implemented for {node_type}")


def _attrs(node):
    out = {}
    for a in node.attribute:
        if a.type == 1:  # FLOAT
            out[a.name] = a.f
        elif a.type == 2:  # INT
            out[a.name] = a.i
        elif a.type == 7:  # INTS
            out[a.name] = list(a.ints)
        elif a.type == 3:  # STRING
            out[a.name] = a.s.decode() if isinstance(a.s, bytes) else a.s
        elif a.type == 4:  # TENSOR
            from onnx import numpy_helper
            out[a.name] = numpy_helper.to_array(a.t)
    return out


class ONNXRunner:
    """Execute an ONNX model on the RK3588 NPU with CPU fallback."""

    def __init__(self, model, precision: str = "float16",
                 verbose: bool = False):
        if onnx is None:
            raise RuntimeError("onnx not installed: pip install onnx")
        if isinstance(model, (str, os.PathLike)):
            model = onnx.load(str(model))
        self.model = model
        self.precision = precision
        self.verbose = verbose
        self._npu_calls = 0
        self._cpu_calls = 0

    def _npu_dispatch(self, op, inputs):
        self._npu_calls += 1
        tmp = tempfile.mkdtemp(prefix="npujob_")
        job_path = os.path.join(tmp, "job.pkl")
        out_path = os.path.join(tmp, "out.pkl")
        job = {"op": op, "precision": self.precision, "inputs": inputs}
        with open(job_path, "wb") as f:
            pickle.dump(job, f)
        env = dict(os.environ, PYTHONPATH="/home/poad42/opennpu/src")
        r = subprocess.run(WORKER + [job_path, out_path], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            sys.stderr.write(r.stdout.decode(errors="replace"))
            raise RuntimeError(f"NPU worker for '{op}' exited {r.returncode}")
        with open(out_path, "rb") as f:
            out_arr = pickle.load(f)
        shutil.rmtree(tmp, ignore_errors=True)
        return out_arr

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        g = self.model.graph
        vals: Dict[str, np.ndarray] = {}
        for init in g.initializer:
            vals[init.name] = numpy_helper.to_array(init).astype(np.float32)
        for gi in g.input:
            if gi.name in inputs:
                vals[gi.name] = np.ascontiguousarray(inputs[gi.name],
                                                    dtype=np.float32)
        for node in g.node:
            outs = self._exec_node(node, vals)
            for name, arr in zip(node.output, outs):
                vals[name] = arr
        result = {}
        for go in g.output:
            result[go.name] = vals[go.name]
        if self.verbose:
            sys.stderr.write(f"[onnx_runner] npu={self._npu_calls} "
                             f"cpu={self._cpu_calls}\n")
        return result

    def _exec_node(self, node, vals):
        nt = node.op_type
        ins = [vals[n] for n in node.input if n]
        attrs = _attrs(node)
        spec = NPU_OPS.get(nt)
        use_npu = False
        npu_op = None
        # Elementwise ops (Add/Sub/Mul/Div) now host-tile to ANY shape with equal
        # element counts, so always use the NPU for them (no shape gate).
        EW = {"Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div"}
        if nt in EW and len(ins) == 2 and ins[0].shape == ins[1].shape:
            use_npu = True
            npu_op = EW[nt]
        # MatMul: dispatch by (X, W) shape to one of the 5 captured templates.
        if nt == "MatMul" and len(ins) == 2:
            xs = tuple(ins[0].shape); ws = tuple(ins[1].shape)
            MM = {((1,64,768),(768,3072)), ((1,64,768),(768,768)),
                  ((1,64,3072),(3072,768)), ((1,64,768),(768,64)),
                  ((1,64,64),(64,768))}
            if (xs, ws) in MM:
                use_npu = True
                npu_op = "matmul"
        # Softmax: dispatch by input shape -- [1,64,768] (captured) or
        # [1,64,64] (attention score axis, captured as softmax64).
        if nt == "Softmax" and len(ins) == 1:
            shp = _shape(ins[0])
            if shp == (1,64,768):
                use_npu = True; npu_op = "softmax"
            elif shp == (1,64,64):
                use_npu = True; npu_op = "softmax64"
        # GELU: native NPU exGelu (fused erf-decomposition), elementwise so it
        # tiles to any shape via the captured (1,64,768) template.
        if nt == "Gelu" and len(ins) == 1:
            use_npu = True; npu_op = "gelu"
        # LayerNormalization: native NPU op (mean/var/sqrt/div/affine in one
        # submit). scale/B are BAKED into the captured template (scattered in
        # scratch, not a patchable io BO), so a TRAINED LM needs one captured
        # template per distinct (scale, B). Dispatch by md5(scale||B) -> a
        # template dir /home/poad42/templates/layernorm_<key>. The default
        # (scale=1,B=0) template is 'layernorm'. Falls back to CPU if no
        # template matches the layer's weights.
        if nt == "LayerNormalization" and len(ins) >= 1 and _shape(ins[0]) == (1,64,768):
            import hashlib
            sc = ins[1] if len(ins) > 1 else np.ones(768, dtype=np.float32)
            bb = ins[2] if len(ins) > 2 else np.zeros(768, dtype=np.float32)
            sca = np.ascontiguousarray(sc, dtype=np.float32).ravel()
            bba = np.ascontiguousarray(bb, dtype=np.float32).ravel()
            key = hashlib.md5(sca.tobytes() + bba.tobytes()).hexdigest()[:8]
            tdir = os.path.join("/home/poad42/templates", f"layernorm_{key}")
            if os.path.isdir(tdir):
                use_npu = True; npu_op = f"layernorm_{key}"
            elif np.allclose(sca, 1.0) and np.allclose(bba, 0.0):
                use_npu = True; npu_op = "layernorm"
        # Elementwise activations (Relu/Sigmoid/Tanh/LeakyRelu) host-tile to ANY
        # shape via the captured (1,64,768) kernel (pjrt template tiling), so use
        # the NPU for them at any single-input shape (no shape gate).
        ACT = {"Relu": "relu", "Sigmoid": "sigmoid",
               "Tanh": "tanh", "LeakyRelu": "leakyrelu"}
        if nt in ACT and len(ins) == 1:
            if nt == "LeakyRelu" and attrs.get("alpha", 0.01) != 0.01:
                pass  # alpha mismatch -> CPU fallback
            else:
                use_npu = True
                npu_op = ACT[nt]
        if spec is not None:
            npu_op2, n_in, in_shapes, _out_shape = spec
            if len(ins) == n_in and all(_shape(t) == s for t, s in zip(ins, in_shapes)):
                use_npu = True
                npu_op = npu_op2
                # extra attribute gates
                if nt == "Transpose" and attrs.get("perm", [0, 2, 1]) != [0, 2, 1]:
                    use_npu = False
                if nt == "Concat" and attrs.get("axis", 1) != 1:
                    use_npu = False
                if nt == "Softmax" and attrs.get("axis", -1) not in (-1, 2, len(in_shapes[0]) - 1):
                    use_npu = False
                if nt == "LeakyRelu" and attrs.get("alpha", 0.01) != 0.01:
                    use_npu = False
        if use_npu:
            if self.verbose:
                sys.stderr.write(f"[onnx_runner] NPU  {nt} -> {npu_op}\n")
            npu_inputs = [ins[0]] if nt == "LayerNormalization" else ins
            return [self._npu_dispatch(npu_op, npu_inputs)]
        # CPU fallback
        self._cpu_calls += 1
        if self.verbose:
            sys.stderr.write(f"[onnx_runner] CPU  {nt}\n")
        return [_cpu_op(nt, attrs, ins)]