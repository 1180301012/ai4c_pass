"""
LayerNorm pass – same single-output pattern as FuseAddLayerNorm_add_first.
Acts as a fallback covering any remaining graphs.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern  –  match layer_norm (single observable output)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_4 = torch.nn.functional.layer_norm(in_2, (1024,), in_1, in_0, 1e-05)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel – same as FuseAddLayerNorm_add_first
# ---------------------------------------------------------------------------
@triton.jit
def _layernorm_kernel_b(
    x_ptr, w_ptr, b_ptr, out_ptr,
    eps,
    IS_BF16: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    off  = row * N + cols

    x = tl.load(x_ptr + off).to(tl.float32)

    x_sum   = tl.sum(x,     axis=0)
    x2_sum  = tl.sum(x * x, axis=0)
    mean    = x_sum  / N
    var     = x2_sum / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + cols).to(tl.float32)
    b = tl.load(b_ptr + cols).to(tl.float32)
    out = (x - mean) * inv_std * w + b

    if IS_BF16:
        tl.store(out_ptr + off, out.to(tl.bfloat16))
    else:
        tl.store(out_ptr + off, out.to(tl.float16))


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap) – single output
# ---------------------------------------------------------------------------
_out_cache_b = {}

@torch.fx.wrap
def triton_layernorm_b(in_0, in_1, in_2):
    """
    in_0 : bias   [1024]
    in_1 : weight [1024]
    in_2 : input  [*, 1024]
    Returns layer_norm(in_2, (1024,), in_1, in_0, 1e-5)
    """
    N       = 1024
    n_rows  = in_2.numel() // N
    is_bf16 = (in_2.dtype == torch.bfloat16)

    # Shape-based key: (shape[0], shape[1], N, is_bf16) uniquely identifies
    shape_key = (in_2.shape[0], in_2.shape[1], N, is_bf16)
    if shape_key not in _out_cache_b:
        _out_cache_b[shape_key] = torch.empty_like(in_2)
    out = _out_cache_b[shape_key]

    _layernorm_kernel_b[(n_rows,)](
        in_2, in_1, in_0, out,
        eps=1e-5,
        IS_BF16=is_bf16,
        N=N,
        BLOCK_SIZE=N,
        num_warps=8,
    )
    return out


# ---------------------------------------------------------------------------
# Replacement entry-point  (IDENTICAL to FuseAddLayerNorm_add_first)
# ---------------------------------------------------------------------------
def replacement_func():
    return triton_layernorm_b