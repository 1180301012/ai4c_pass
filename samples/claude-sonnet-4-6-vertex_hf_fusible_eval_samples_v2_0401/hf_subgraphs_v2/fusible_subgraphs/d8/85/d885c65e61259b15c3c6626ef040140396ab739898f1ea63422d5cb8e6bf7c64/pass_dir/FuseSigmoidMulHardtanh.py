"""
Fuse: sigmoid(conv_out) * x  then clamp to [0, 6]

Pattern matched:
    s   = conv_out.sigmoid()
    m   = x * s               (broadcast: conv_out is [B,C,1,1], x is [B,C,H,W])
    out = hardtanh(m, 0.0, 6.0, inplace=False)

Strategy:
  - For N >= 2M elements (B=32 cases): 2-D grid Triton kernel.
    Reads x once and writes once (vs. 3 separate kernel passes in eager).
    SE sigmoid computed once per (b,c) pair in each spatial tile program.
  - For N < 2M elements (B=1 cases): tensor-method fallback.
    Triton's per-kernel dispatch overhead exceeds the bandwidth savings
    for these small workloads.
"""

import torch
import triton
import triton.language as tl


# ------------------------------------------------------------------ #
#  Pattern  (must mirror model.py dataflow exactly)                   #
# ------------------------------------------------------------------ #
def pattern(conv_out, x):
    s = conv_out.sigmoid()
    m = x * s
    r = torch.nn.functional.hardtanh(m, 0.0, 6.0, False)
    return r


def replacement_args(conv_out, x):
    return (conv_out, x)


# ------------------------------------------------------------------ #
#  Triton kernel  –  2-D grid  (axis-0: bc pair, axis-1: hw tile)    #
# ------------------------------------------------------------------ #
@triton.autotune(
    configs=[
        # baseline (stages=1)
        triton.Config({"BLOCK_SIZE": 256},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        # pipeline depth 2 – hides memory latency, good for bf16
        triton.Config({"BLOCK_SIZE": 256},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
    ],
    key=["HW"],  # key on spatial size; BC is always large (7296) for Triton path
)
@triton.jit
def _fused_se_2d_kernel(
    se_ptr,               # [B*C]   – flattened conv_out [B,C,1,1]
    x_ptr,                # [B*C*HW]
    out_ptr,              # [B*C*HW]
    HW,                   # H * W  (runtime int, autotune key)
    BLOCK_SIZE: tl.constexpr,
):
    bc_id    = tl.program_id(0)   # which (b,c) pair
    hw_block = tl.program_id(1)   # which spatial tile

    # Load SE value and apply sigmoid in fp32 for numerical stability
    se_val     = tl.load(se_ptr + bc_id)
    se_sig_f32 = tl.sigmoid(se_val.to(tl.float32))

    # Spatial tile
    hw_start = hw_block * BLOCK_SIZE
    hw_offs  = hw_start + tl.arange(0, BLOCK_SIZE)
    mask     = hw_offs < HW

    x_offs = bc_id * HW + hw_offs
    x_val  = tl.load(x_ptr + x_offs, mask=mask, other=0.0)

    se_sig = se_sig_f32.to(x_val.dtype)
    out    = x_val * se_sig
    out    = tl.minimum(tl.maximum(out, 0.0), 6.0)

    tl.store(out_ptr + x_offs, out, mask=mask)


# ------------------------------------------------------------------ #
#  Kernel wrapper  (must be @torch.fx.wrap)                           #
# ------------------------------------------------------------------ #

# Empirical crossover: Triton dispatch overhead pays off only for large
# tensors (B=32). For B=1 (N ~ 178K–525K), the per-invocation overhead
# exceeds the bandwidth savings; we use tensor-method ops instead.
_TRITON_THRESHOLD = 2_000_000   # elements


@torch.fx.wrap
def fused_se_sigmoid_mul_hardtanh(conv_out, x):
    """
    conv_out : [B, C, 1, 1]
    x        : [B, C, H, W]
    returns  : [B, C, H, W]  ==  clamp(x * sigmoid(conv_out), 0, 6)
    """
    # ---- small tensors: avoid Triton overhead, use tensor methods ----
    N = x.numel()
    if N < _TRITON_THRESHOLD:
        s = conv_out.sigmoid()      # [B,C,1,1] – tiny, fast
        m = x * s                   # broadcast  → [B,C,H,W]
        return m.clamp(0.0, 6.0)   # == hardtanh(·, 0, 6)

    # ---- large tensors: 2-D grid Triton saves ~2× memory bandwidth ----
    # Use numel() to avoid multiple Python shape-attribute accesses
    BC = conv_out.numel()   # B*C*1*1 = B*C
    HW = N // BC            # H*W  (integer div of two cached ints)

    # conv_out is conv2d output (always contiguous); x is model input (always contiguous).
    # Use view() directly to skip the redundant contiguity check in .contiguous().
    se_flat  = conv_out.view(BC)
    x_flat   = x.view(N)
    out_flat = torch.empty_like(x_flat)

    grid = lambda META: (BC, triton.cdiv(HW, META["BLOCK_SIZE"]))

    _fused_se_2d_kernel[grid](
        se_flat,
        x_flat,
        out_flat,
        HW,
    )

    return out_flat.view_as(x)   # reshape without extra shape lookups


# ------------------------------------------------------------------ #
#  Replacement  (zero-arg, returns callable)                          #
# ------------------------------------------------------------------ #
def replacement_func():
    return fused_se_sigmoid_mul_hardtanh