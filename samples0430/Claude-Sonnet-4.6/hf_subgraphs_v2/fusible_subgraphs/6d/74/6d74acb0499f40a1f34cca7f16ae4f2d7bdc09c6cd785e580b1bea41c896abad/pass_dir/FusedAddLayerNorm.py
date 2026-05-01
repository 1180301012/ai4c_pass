import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────
# Pattern: exactly mirrors the model.py computation graph
# ─────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3):
    tmp_3 = in_3 + in_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    return tmp_15


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ─────────────────────────────────────────────────────────
# Optimised kernel for N = 768  (= 3 × 256, zero masking)
#
# N=768 splits evenly into three 256-element chunks, so
# every tl.arange call is exactly 256 elements and no
# lanes are wasted on masking (vs. 25% waste at 1024).
# ─────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['M'],
)
@triton.jit
def _fused_ln_n768(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr, out_ptr,
    M, N, eps,
):
    row  = tl.program_id(0)
    base = row * N          # N = 768

    # Integer literals → tl.arange gets compile-time-constant bounds
    o0 = tl.arange(0, 256)
    o1 = 256 + tl.arange(0, 256)
    o2 = 512 + tl.arange(0, 256)

    # ── Residual-add + cast to fp32 (3 mask-free chunks) ─
    x0 = (tl.load(in2_ptr + base + o0, eviction_policy='evict_first').to(tl.float32) +
          tl.load(in3_ptr + base + o0, eviction_policy='evict_first').to(tl.float32))
    x1 = (tl.load(in2_ptr + base + o1, eviction_policy='evict_first').to(tl.float32) +
          tl.load(in3_ptr + base + o1, eviction_policy='evict_first').to(tl.float32))
    x2 = (tl.load(in2_ptr + base + o2, eviction_policy='evict_first').to(tl.float32) +
          tl.load(in3_ptr + base + o2, eviction_policy='evict_first').to(tl.float32))

    # ── Mean ─────────────────────────────────────────────
    mean = (tl.sum(x0, axis=0) + tl.sum(x1, axis=0) + tl.sum(x2, axis=0)) / N

    # ── Centred values ────────────────────────────────────
    d0 = x0 - mean
    d1 = x1 - mean
    d2 = x2 - mean

    # ── Variance + rstd ───────────────────────────────────
    var  = (tl.sum(d0 * d0, axis=0) + tl.sum(d1 * d1, axis=0) +
            tl.sum(d2 * d2, axis=0)) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # ── Scale + shift (weight / bias stay in cache) ───────
    w0 = tl.load(in1_ptr + o0).to(tl.float32)
    w1 = tl.load(in1_ptr + o1).to(tl.float32)
    w2 = tl.load(in1_ptr + o2).to(tl.float32)
    b0 = tl.load(in0_ptr + o0).to(tl.float32)
    b1 = tl.load(in0_ptr + o1).to(tl.float32)
    b2 = tl.load(in0_ptr + o2).to(tl.float32)

    tl.store(out_ptr + base + o0, w0 * (d0 * rstd) + b0)
    tl.store(out_ptr + base + o1, w1 * (d1 * rstd) + b1)
    tl.store(out_ptr + base + o2, w2 * (d2 * rstd) + b2)


# ─────────────────────────────────────────────────────────
# General fallback kernel  (any N ≤ 1024, uses masking)
# ─────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['M', 'N'],
)
@triton.jit
def _fused_ln_general(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr, out_ptr,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row     = tl.program_id(0)
    base    = row * N
    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N

    x2 = tl.load(in2_ptr + base + offsets, mask=mask, other=0.0,
                  eviction_policy='evict_first').to(tl.float32)
    x3 = tl.load(in3_ptr + base + offsets, mask=mask, other=0.0,
                  eviction_policy='evict_first').to(tl.float32)
    x  = x2 + x3

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    norm = diff * rstd

    w   = tl.load(in1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b   = tl.load(in0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + base + offsets, w * norm + b, mask=mask)


# ─────────────────────────────────────────────────────────
# Wrapper — @torch.fx.wrap keeps FX from tracing inside
# ─────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_add_layernorm(in_0, in_1, in_2, in_3):
    N   = in_2.shape[-1]
    M   = in_2.numel() // N
    out = torch.empty(in_2.shape, dtype=torch.float32, device=in_2.device)

    if N == 768:
        # Fast path: 3 × 256 chunks, zero masking
        _fused_ln_n768[(M,)](
            in_0, in_1, in_2, in_3, out,
            M=M, N=N, eps=1e-7,
        )
    else:
        # General path: single chunk + mask
        _fused_ln_general[(M,)](
            in_0, in_1, in_2, in_3, out,
            M=M, N=N, eps=1e-7,
        )

    return out


def replacement_func():
    return fused_add_layernorm