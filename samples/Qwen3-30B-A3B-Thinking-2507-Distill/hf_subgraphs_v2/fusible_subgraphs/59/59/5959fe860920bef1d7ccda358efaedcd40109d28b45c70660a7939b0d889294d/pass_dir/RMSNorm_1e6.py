import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: fuse the full RMSNorm pipeline
#   x → to(float32) → pow(2) → mean(-1) → +eps → rsqrt → multiply → to(bf16)
#   then element-wise multiply by weight
# Input x : [B, S, H] bfloat16   (in_2)
# Input w : [H]       bfloat16   (in_0 / weight)
# Output  : [B, S, H] bfloat16
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['H'],
)
@triton.jit
def rms_norm_1e6_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    H: tl.constexpr,          # hidden dim (always 2048 in this workload)
    BLOCK_SIZE: tl.constexpr,
):
    row_id  = tl.program_id(0)
    row_ptr = x_ptr + row_id * H
    cols    = tl.arange(0, BLOCK_SIZE)
    mask    = cols < H

    # Load input (bfloat16 → float32)
    x = tl.load(row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # RMS  (sum of squares, then rsqrt)
    mean_sq = tl.sum(x * x, axis=0) / H
    rstd    = tl.math.rsqrt(mean_sq + 1e-6)

    # Load weight (bfloat16 → float32) and normalise
    w   = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    out = (x * rstd) * w

    tl.store(out_ptr + row_id * H + cols, out.to(tl.bfloat16), mask=mask)


@torch.fx.wrap
def rms_norm_1e6(x, weight):
    """
    Fused RMSNorm(bf16, eps=1e-6):
      x      : [B, S, H] bfloat16   (token embeddings)
      weight : [H]       bfloat16   (input layernorm weight)
    Returns: [B, S, H] bfloat16
    """
    H   = x.shape[-1]          # 2048
    N   = x.numel() // H       # B * S
    out = torch.empty_like(x)

    rms_norm_1e6_kernel[(N,)](
        x, weight, out,
        H,
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Pattern / replacement glue
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_2, in_0):
    tmp_10 = in_2.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_17 = in_0 * tmp_16
    return tmp_17


def replacement_args(in_2, in_0):
    return (in_2, in_0)


def replacement_func():
    return rms_norm_1e6