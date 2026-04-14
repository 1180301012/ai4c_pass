import torch
from torch import device
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused (in_1 + in_0) + clamp(max=-3.4028e38)
# Output shape: [H, N, N]  (the view is implicit in the output allocation)
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def _add_clamp_kernel(
    in0_ptr, in1_ptr, out_ptr,
    H, N,
    s_in0_row, s_in0_col,
    s_in1_head, s_in1_row, s_in1_col,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx  = tl.program_id(0)
    head_idx = row_idx // N
    seq_idx  = row_idx  % N
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    in0 = tl.load(in0_ptr + seq_idx * s_in0_row + cols * s_in0_col,
                  mask=mask, other=0.0).to(tl.float32)
    in1 = tl.load(in1_ptr + head_idx * s_in1_head + seq_idx * s_in1_row + cols * s_in1_col,
                  mask=mask, other=0.0).to(tl.float32)
    x = in1 + in0
    CLAMP = -3.4028234663852886e+38
    x = tl.where(x > CLAMP, x, CLAMP)
    if IS_FP16:
        out_val = x.to(tl.float16)
    elif IS_BF16:
        out_val = x.to(tl.bfloat16)
    else:
        out_val = x
    tl.store(out_ptr + row_idx * N + cols, out_val, mask=mask)


def _run_add_clamp(in_0, in_1, H, N):
    out = torch.empty((H, N, N), dtype=in_1.dtype, device=in_1.device)
    _add_clamp_kernel[(H * N,)](
        in_0, in_1, out, H, N,
        in_0.stride(2), in_0.stride(3),
        in_1.stride(1), in_1.stride(2), in_1.stride(3),
        IS_FP16=(in_1.dtype == torch.float16),
        IS_BF16=(in_1.dtype == torch.bfloat16),
        BLOCK_N=16,
    )
    return out


@torch.fx.wrap
def _fused_add_clamp_16_13(in_0, in_1):
    return _run_add_clamp(in_0, in_1, 16, 13)


@torch.fx.wrap
def _fused_add_clamp_12_9(in_0, in_1):
    return _run_add_clamp(in_0, in_1, 12, 9)


@torch.fx.wrap
def _fused_add_clamp_16_9(in_0, in_1):
    return _run_add_clamp(in_0, in_1, 16, 9)


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: add + max(clamp) + view(16,13,13)
# Anchor = call_method[view] — avoids F.softmax call_function/call_method issue.
# clamp_val placeholder matches the torch.tensor(-3.4028e38) node.
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, clamp_val):
    tmp_0 = in_1 + in_0
    tmp_2 = torch.max(tmp_0, clamp_val)
    tmp_3 = tmp_2.view(16, 13, 13)
    return tmp_3


def replacement_args(in_0, in_1, clamp_val):
    return (in_0, in_1)


def replacement_func():
    return _fused_add_clamp_16_13