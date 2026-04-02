"""
Pass: Triton-fused layer_norm  (normalized_shape=(1024,), eps=1e-05)
Matches the standalone layer_norm in all three graphs and replaces it with a
fast Triton kernel.

Pattern returns a SINGLE tensor → no multi-output assertion issue.
Grid: (T,) programs, one per time step. Each program reduces C=1024 elements.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern  — matches layer_norm in all three graphs
# ---------------------------------------------------------------------------
def pattern(x, in_1, in_0):
    return torch.nn.functional.layer_norm(x, (1024,), in_1, in_0, 1e-05)


def replacement_args(x, in_1, in_0):
    return (x, in_1, in_0)


# ---------------------------------------------------------------------------
# Triton layer-norm kernel  (dtype-agnostic)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_C': 1024}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_C': 1024}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_C': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 1024}, num_warps=8, num_stages=2),
    ],
    key=['T', 'C'],
)
@triton.jit
def _layernorm_kernel(
    x_ptr,     x_stride_t,     # input  [1, T, C], stride_t = C = 1024
    out_ptr,   out_stride_t,   # output [1, T, C], stride_t = C = 1024
    w_ptr,                     # weight [C]
    b_ptr,                     # bias   [C]
    T, C,
    BLOCK_C: tl.constexpr,
):
    """One program per time step."""
    t = tl.program_id(0)
    c_idx = tl.arange(0, BLOCK_C)

    # ---- load row x[t, :] ----
    raw = tl.load(x_ptr + t * x_stride_t + c_idx)
    xf = raw.to(tl.float32)

    # ---- mean ----
    mean = tl.sum(xf, axis=0) / C

    # ---- variance ----
    diff = xf - mean
    var  = tl.sum(diff * diff, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    # ---- normalize + scale + shift ----
    w = tl.load(w_ptr + c_idx).to(tl.float32)
    b = tl.load(b_ptr + c_idx).to(tl.float32)
    out = diff * rstd * w + b

    # ---- store in original dtype ----
    tl.store(out_ptr + t * out_stride_t + c_idx, out.to(raw.dtype))


# ---------------------------------------------------------------------------
# Wrapper  (@torch.fx.wrap → single opaque FX node)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_layer_norm(x, in_1, in_0):
    """
    x    : [1, T, C]  input tensor (any float dtype)
    in_1 : [C]        weight
    in_0 : [C]        bias
    Returns: [1, T, C] layer-normed output
    """
    _B, T, C = x.shape          # [1, 249, 1024]

    out = torch.empty_like(x)

    _layernorm_kernel[(T,)](
        x,   x.stride(1),
        out, out.stride(1),
        in_1, in_0,
        T, C,
    )

    return out


# ---------------------------------------------------------------------------
# Replacement entry point
# ---------------------------------------------------------------------------
def replacement_func():
    return triton_layer_norm