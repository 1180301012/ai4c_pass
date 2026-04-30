import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N, eps,
    IS_FP16: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per row. Fused layer-norm with float32 accumulation."""
    row = tl.program_id(0)
    row_start = row * N
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Load and upcast to float32 for numerical precision
    x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)

    # Mean and variance (two-pass, all float32)
    mean = tl.sum(x, axis=0) / N
    d = x - mean
    var  = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = d * rstd

    # Affine transform
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr   + cols, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * w + b

    # Store back in the original dtype
    if IS_FP16:
        tl.store(out_ptr + row_start + cols, out.to(tl.float16), mask=mask)
    else:
        tl.store(out_ptr + row_start + cols, out.to(tl.bfloat16), mask=mask)


# Module-level cache: avoids recomputing shape-derived values on every call.
_layernorm_cache = {}


@torch.fx.wrap
def _triton_layernorm_wrapper(in_0, in_1, tmp_3):
    """
    in_0 : bias   [N]
    in_1 : weight [N]
    tmp_3: input  [..., N]  (the add+reshape result in the original graph)
    Returns layer_norm(tmp_3, weight=in_1, bias=in_0).
    """
    # Cache key: shape + dtype to avoid per-call Python overhead
    key = (tmp_3.shape[-1], tmp_3.dtype)
    if key not in _layernorm_cache:
        N = tmp_3.shape[-1]
        M = tmp_3.numel() // N
        is_fp16 = 1 if tmp_3.dtype == torch.float16 else 0
        BLOCK_N = triton.next_power_of_2(N)
        _layernorm_cache[key] = (N, M, is_fp16, BLOCK_N)
    N, M, is_fp16, BLOCK_N = _layernorm_cache[key]

    out = torch.empty_like(tmp_3)

    _layernorm_kernel[(M,)](
        tmp_3, in_1, in_0, out,
        M=M, N=N, eps=1e-05,
        IS_FP16=is_fp16,
        BLOCK_N=BLOCK_N,
        num_warps=1,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern: match layer_norm only (single output → no multi-output crash).
# tmp_3 is the output of the preceding add+reshape; those stay in the graph.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, tmp_3):
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-05)
    return tmp_4


def replacement_args(in_0, in_1, tmp_3):
    return (in_0, in_1, tmp_3)


def replacement_func():
    return _triton_layernorm_wrapper