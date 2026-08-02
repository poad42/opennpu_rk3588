import jax
import numpy as np
from jax._src import core as core
from jax._src.interpreters import mlir

npu_mm_p = core.Primitive("npu_cached_mm")
npu_mm_p.multiple_results = False

_WT_N = {}

def register_weight_n(w_idx, n):
    _WT_N[w_idx] = n

def _abstract(x, *, w_idx, out_n=None):
    out_n = out_n or _WT_N.get(w_idx, x.shape[-1])
    out_shape = list(x.shape)
    out_shape[-1] = out_n
    return core.ShapedArray(tuple(out_shape), x.dtype)


npu_mm_p.def_abstract_eval(_abstract)


def _lower(ctx, x, *, w_idx, out_n=None):
    aval = ctx.avals_in[0]
    out_n = out_n or _WT_N.get(w_idx, aval.shape[-1])
    out_shape = list(aval.shape)
    out_shape[-1] = out_n
    result_ty = mlir.ir.RankedTensorType.get(out_shape, mlir.ir.F16Type.get())
    op = mlir.custom_call(
        "npu_cached_mm",
        result_types=[result_ty],
        operands=[x],
        backend_config=f"w_idx={w_idx}",
        has_side_effect=False,
    )
    return [op.result]


mlir.register_lowering(npu_mm_p, _lower)


def npu_cached_mm(x, w_idx, out_n=None):
    return npu_mm_p.bind(x, w_idx=w_idx, out_n=out_n)


def write_weights(weights, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    for i, w in enumerate(weights):
        np.ascontiguousarray(w.astype(np.float16)).tofile(f"{out_dir}/w{i}.bin")