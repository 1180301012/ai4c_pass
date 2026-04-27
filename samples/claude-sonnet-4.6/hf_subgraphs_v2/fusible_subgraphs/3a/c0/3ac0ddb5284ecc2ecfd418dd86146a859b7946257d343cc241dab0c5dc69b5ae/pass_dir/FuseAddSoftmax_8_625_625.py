import torch
import triton
import triton.language as tl


# Fused broadcast-add + softmax kernel for shape [1,8,625,625]
# Each program handles one row of 625 elements
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=[],
)
@triton.jit
def _fused_add_softmax_kernel_625_625(
    in0_ptr,   # [1, 1, 625, 625] broadcast attention mask
    in1_ptr,   # [1, 8, 625, 625] attention scores
    out_ptr,   # [1, 8, 625, 625] softmax output
):
    # Constants for this kernel specialization
    H = 625
    W = 625
    BLOCK_W = 1024  # smallest power-of-2 >= 625

    row = tl.program_id(0)
    head = row // H
    pos  = row %  H

    # Pointer bases (contiguous layout)
    in0_base = in0_ptr + pos * W                  # in_0[0, 0,    pos, :]
    in1_base = in1_ptr + (head * H + pos) * W     # in_1[0, head, pos, :]
    out_base = out_ptr + (head * H + pos) * W     # out [0, head, pos, :]

    cols = tl.arange(0, BLOCK_W)
    mask = cols < W

    # Load inputs in native dtype, promote to float32 for stable softmax
    a = tl.load(in0_base + cols, mask=mask, other=0.0)
    b = tl.load(in1_base + cols, mask=mask, other=0.0)

    x = a.to(tl.float32) + b.to(tl.float32)

    # Set padding lanes to -inf so they don't affect max / sum
    x = tl.where(mask, x, float('-inf'))

    # Online softmax: subtract max for numerical stability
    x_max    = tl.max(x, axis=0)
    x_shift  = x - x_max
    x_exp    = tl.exp(x_shift)
    x_exp    = tl.where(mask, x_exp, 0.0)   # zero out padding
    x_sum    = tl.sum(x_exp, axis=0)
    out_val  = x_exp / x_sum

    # Store back in original dtype
    tl.store(out_base + cols, out_val.to(a.dtype), mask=mask)


def pattern(in_0, in_1):
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.view(8, 625, 625)
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.view(1, 8, 625, 625)
    tmp_4 = tmp_3.view(8, 625, 625)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.0, training=False)
    return (tmp_5, tmp_3)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@torch.fx.wrap
def _fused_add_softmax_625_625_wrapper(in_0, in_1):
    N, H, W = 8, 625, 625
    # Allocate output as [1, N, H, W]
    out = torch.empty((1, N, H, W), dtype=in_0.dtype, device=in_0.device)
    N_ROWS = N * H  # 5000 programs

    _fused_add_softmax_kernel_625_625[(N_ROWS,)](
        in_0,
        in_1,
        out,
    )

    # tmp_3 = [1, 8, 625, 625], tmp_5 = [8, 625, 625]  (shared storage, no copy)
    tmp_3 = out
    tmp_5 = out.view(N, H, W)
    return (tmp_5, tmp_3)


def replacement_func():
    return _fused_add_softmax_625_625_wrapper