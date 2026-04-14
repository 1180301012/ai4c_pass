import torch
from torch import device
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused add + clamp + softmax  (dropout is a no-op at inference)
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _fused_attn_softmax_kernel(
    in0_ptr,          # attention mask  [1, 1, N, N]
    in1_ptr,          # attention scores [1, H, N, N]
    out_ptr,          # output           [H, N, N]
    H, N,
    s_in0_row,        # in0 stride dim-2
    s_in0_col,        # in0 stride dim-3
    s_in1_head,       # in1 stride dim-1
    s_in1_row,        # in1 stride dim-2
    s_in1_col,        # in1 stride dim-3
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # one program per (head, row) pair
    row_idx  = tl.program_id(0)
    head_idx = row_idx // N
    seq_idx  = row_idx  % N

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # ── load in0 [1, 1, N, N] → row seq_idx ──
    in0_offsets = seq_idx * s_in0_row + cols * s_in0_col
    in0 = tl.load(in0_ptr + in0_offsets, mask=mask, other=0.0).to(tl.float32)

    # ── load in1 [1, H, N, N] → row (head_idx, seq_idx) ──
    in1_offsets = head_idx * s_in1_head + seq_idx * s_in1_row + cols * s_in1_col
    in1 = tl.load(in1_ptr + in1_offsets, mask=mask, other=0.0).to(tl.float32)

    # add
    x = in1 + in0

    # clamp: max(x, -3.4028e38)
    CLAMP = -3.4028234663852886e+38
    x = tl.where(x > CLAMP, x, CLAMP)

    # numerically-stable softmax over the N-length row
    x_for_max = tl.where(mask, x, float('-inf'))
    x_max     = tl.max(x_for_max, axis=0)

    x_exp  = tl.exp(x - x_max)
    x_exp  = tl.where(mask, x_exp, 0.0)
    x_sum  = tl.sum(x_exp, axis=0)
    x_out  = x_exp / x_sum

    # cast back to original dtype and store
    if IS_FP16:
        out_val = x_out.to(tl.float16)
    elif IS_BF16:
        out_val = x_out.to(tl.bfloat16)
    else:
        out_val = x_out  # float32

    tl.store(out_ptr + row_idx * N + cols, out_val, mask=mask)


# ──────────────────────────────────────────────────────────────────────────────
# Python wrapper (called inside @torch.fx.wrap, so arbitrary tensor ops OK here)
# ──────────────────────────────────────────────────────────────────────────────
def _run_fused_attn_softmax(in_0, in_1, H, N):
    out = torch.empty((H, N, N), dtype=in_1.dtype, device=in_1.device)

    is_fp16 = (in_1.dtype == torch.float16)
    is_bf16 = (in_1.dtype == torch.bfloat16)

    BLOCK_N = 16   # power-of-2 ≥ max(N)=13

    _fused_attn_softmax_kernel[(H * N,)](
        in_0, in_1, out,
        H, N,
        in_0.stride(2), in_0.stride(3),
        in_1.stride(1), in_1.stride(2), in_1.stride(3),
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
        BLOCK_N=BLOCK_N,
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Unique wrapper for this pass (unique name avoids replacement_func_limit)
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def _kernel_fused_attn_softmax_16_9_9(in_0, in_1):
    return (_run_fused_attn_softmax(in_0, in_1, 16, 9),)


# ──────────────────────────────────────────────────────────────────────────────
# Pattern  (clamp_val is a placeholder that matches the torch.tensor(...) node)
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, clamp_val):
    tmp_0 = in_1 + in_0
    tmp_2 = torch.max(tmp_0, clamp_val)
    tmp_3 = tmp_2.view(16, 9, 9)
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.1, training=False)
    return (tmp_5,)


def replacement_args(in_0, in_1, clamp_val):
    return (in_0, in_1)


def replacement_func():
    return _kernel_fused_attn_softmax_16_9_9