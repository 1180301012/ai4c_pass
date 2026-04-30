"""
Fast LayerNorm pass (single-output).
Optimized with autotune, constexpr N, and single-pass variance.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=['M'],
)
@triton.jit
def _triton_layernorm_kernel_b(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    row_off = row * N

    # Load input row
    x = tl.load(x_ptr + row_off + offsets)

    # Upcast to fp32
    x_f32 = x.to(tl.float32)

    # Single-pass mean + variance using E[x^2] - E[x]^2
    # (numerically fine for float16/bfloat16 inputs accumulated in fp32)
    sum_x  = tl.sum(x_f32,       0)
    sum_x2 = tl.sum(x_f32 * x_f32, 0)
    mean = sum_x / N
    var  = sum_x2 / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    # Affine transform
    w     = tl.load(w_ptr + offsets).to(tl.float32)
    b_val = tl.load(b_ptr + offsets).to(tl.float32)
    out   = w * (x_f32 - mean) * rstd + b_val

    tl.store(out_ptr + row_off + offsets, out.to(x.dtype))


@torch.fx.wrap
def triton_layernorm_b(in_0, in_1, tmp_2):
    """
    Triton fast layer-norm.

    in_0  : bias   [N]
    in_1  : weight [N]
    tmp_2 : input  [*, N]

    Returns: ln_out
    """
    N = tmp_2.shape[-1]          # always 1024
    M = tmp_2.numel() // N       # number of rows

    out = torch.empty_like(tmp_2)

    _triton_layernorm_kernel_b[(M,)](
        tmp_2, in_1, in_0, out,
        M,
        N=N,
        BLOCK_SIZE=1024,
    )

    return out


# ── Pattern & replacement hooks ──────────────────────────────────────────────

def pattern(in_0, in_1, tmp_2):
    tmp_4 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return tmp_4


def replacement_args(in_0, in_1, tmp_2):
    return (in_0, in_1, tmp_2)


def replacement_func():
    return triton_layernorm_b