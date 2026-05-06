"""
Shared Triton kernels for rotary mask/attention operations.
Imported by individual pass files for each N value.
"""

import torch
import triton
import triton.language as tl

BLOCK_POS = 512
BLOCK_INV = 512

# ============================================================
# Kernel 1: Fused causal + attention mask → tmp_13
# ============================================================
# tmp_13[i,j] = float32(
#     bool(arange(N)[j] <= in_2[j])  AND  bool(in_0[b_idx,i])
# )
# where b_idx = i // N and j_idx = i % N.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    ],
    key=['total', 'N'],
)
@triton.jit
def fused_causal_attn_kernel(
    in0_ptr,   # [B, N] int64 – original attention mask
    in2_ptr,   # [N]    int64   – cache_position
    out_ptr,   # [1,N,N] float32 – output tmp_13
    N,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    j_idx = offsets % N
    i_idx = offsets // N

    bool0    = (tl.load(in0_ptr + i_idx * N + j_idx, mask=mask, other=0) != 0)
    in2_val  = tl.load(in2_ptr + j_idx, mask=mask, other=0)
    success  = (in2_val <= j_idx)

    result   = (tl.cast(bool0, tl.float32) *
                tl.cast(bool(success), tl.float32)).to(tl.float32)
    tl.store(out_ptr + offsets, result, mask=mask)


# ============================================================
# Kernel 2: int64 position_ids → float32 → tmp_22  [1, N]
# ============================================================
# Single block (BLOCK_POS=512 >= max N=512), no masking issue.

@triton.jit
def pos_ids_f32_kernel(
    in3_ptr,   # [1, N] int64
    out_ptr,   # [1, N] float32
    N,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N
    pos     = tl.load(in3_ptr + offsets, mask=mask, other=0).to(tl.float32)
    tl.store(out_ptr + offsets, pos, mask=mask)


# ============================================================
# Kernel 3: inv_freq [N] → [1, N, 1] float32 → tmp_21
# ============================================================

@triton.jit
def inv_freq_shape_kernel(
    in1_ptr,   # [N]    float16/bfloat16/float32
    out_ptr,   # [1,N,1] float32
    N,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N
    vals    = tl.load(in1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, vals, mask=mask)


# ============================================================
# Public dispatchers (wrapped for FX graph replacement)
# ============================================================

def _run_causal_attn(in_0, in_2, N_val):
    """Return tmp_13 [1, N, N] float32."""
    out  = torch.empty((1, N_val, N_val), dtype=torch.float32, device=in_0.device)
    total  = N_val * N_val
    BLOCK_SIZE  = 128
    grid       = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_causal_attn_kernel[grid](in_0, in_2, out, N=N_val, total=total)
    return out


def _run_pos_ids(in_3, N_val):
    """Return tmp_22 [1, N] float32."""
    out   = torch.empty((1, N_val), dtype=torch.float32, device=in_3.device)
    BLOCK_SIZE = 512
    grid       = (1,)
    pos_ids_f32_kernel[grid](in_3, out, N=N_val, BLOCK_SIZE=BLOCK_SIZE)
    return out


def _run_inv_freq(in_1, N_val):
    """Return tmp_21 [1, N, 1] float32."""
    out   = torch.empty((1, N_val, 1), dtype=torch.float32, device=in_1.device)
    BLOCK_SIZE = 512
    grid       = (1,)
    inv_freq_shape_kernel[grid](in_1, out, N=N_val, BLOCK_SIZE=BLOCK_SIZE)
    return out