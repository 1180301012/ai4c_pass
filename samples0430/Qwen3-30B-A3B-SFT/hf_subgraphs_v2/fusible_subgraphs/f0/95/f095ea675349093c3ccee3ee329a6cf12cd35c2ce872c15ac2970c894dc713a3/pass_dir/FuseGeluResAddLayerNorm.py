import torch
import triton
import triton.language as tl


# ── GELU + transpose + residual add  (single output: tmp_8) ─────────────────
# Matches:  slice → gelu → transpose(1,2) → add(res, …)  → tmp_8
# Single output avoids subgraph-rewriter tuple-return crash.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK': 1024}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK': 1024}, num_warps=32, num_stages=3),
    ],
    key=['D'],
)
@triton.jit
def _gelu_t_add_kernel(
    x_ptr,        # [B, C, L]
    res_ptr,      # [B, L, C]
    out_ptr,      # [B, L, C]
    C,
    L,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row    = tl.program_id(0)
    b_dim  = row // L
    l      = row % L

    x_base   = b_dim * C * L + l          # x[b, :, l]
    lr_base  = row * D                    # res[b, l, :]

    cols = tl.arange(0, D)

    x_raw   = tl.load(x_ptr   + x_base   + cols * L)
    x_f32   = x_raw.to(tl.float32)
    # GELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    # tanh(u) = 2*sigmoid(2u) - 1  →  cdf = sigmoid(2u) + sigmoid(2u) - ...
    # cdf = 0.5*(1+tanh(u)) = sigmoid(2u)
    _u = 0.7978845608028654 * (x_f32 + 0.044715 * x_f32 * x_f32 * x_f32)
    cdf   = tl.sigmoid(2.0 * _u)
    gelu_f32 = x_f32 * cdf

    res_raw  = tl.load(res_ptr + lr_base + cols)
    z_f32    = gelu_f32 + res_raw.to(tl.float32)
    tl.store(out_ptr + lr_base + cols, z_f32.to(res_raw.dtype))


@torch.fx.wrap
def fused_gelu_t_add(x, res):
    B, C, L = x.shape
    out = torch.empty_like(res)
    _gelu_t_add_kernel[lambda meta: (B * L,)](x, res, out, C, L, C)
    return out


# ── Pattern / replacement plumbing ────────────────────────────────────────────

def pattern(x, res):
    """
    Matches:  slice → gelu → transpose(1,2) → add(res, …)
    Returns single tmp_8 — no tuple avoids subgraph-rewriter crash.
    """
    tmp_4 = x[:, :, :-1]
    tmp_5 = torch.nn.functional.gelu(tmp_4)
    tmp_6 = tmp_5.transpose(1, 2)
    return res + tmp_6


def replacement_args(x, res):
    return (x, res)


def replacement_func():
    return fused_gelu_t_add