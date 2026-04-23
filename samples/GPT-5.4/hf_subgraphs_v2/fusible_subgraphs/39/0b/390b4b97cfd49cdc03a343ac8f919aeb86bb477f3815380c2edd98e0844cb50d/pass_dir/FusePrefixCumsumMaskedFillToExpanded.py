import torch
import triton
import triton.language as tl
from torch import device


def pattern(in_0, in_1):
    tmp_1 = in_1.cumsum(-1)
    tmp_2 = tmp_1 - 1
    tmp_3 = in_0.__eq__(0)
    tmp_2 = tmp_2.masked_fill_(tmp_3, 1)
    tmp_5 = tmp_2.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    tmp_7 = tmp_6.to(device(type='cuda', index=0))
    return tmp_7


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def _prefix_kernel(
    attention_mask_ptr,
    input_long_ptr,
    tmp2_out_ptr,
    batch_size,
    n_cols,
    attn_row_stride,
    long_row_stride,
    out_row_stride,
    MAX_N: tl.constexpr,
):
    row = tl.program_id(0)
    running = tl.full((), 0, tl.int64)

    for j in range(MAX_N):
        if j < n_cols:
            x = tl.load(input_long_ptr + row * long_row_stride + j)
            running = running + x
            val = running - 1
            m = tl.load(attention_mask_ptr + row * attn_row_stride + j)
            val = tl.where(m == 0, 1, val)
            tl.store(tmp2_out_ptr + row * out_row_stride + j, val)


@torch.fx.wrap
def fused_prefix(in_0, in_1):
    batch_size = in_1.shape[0]
    n_cols = in_1.shape[1]
    tmp2 = torch.empty((batch_size, n_cols), device=in_1.device, dtype=in_1.dtype)

    if n_cols <= 16:
        max_n = 16
    elif n_cols <= 32:
        max_n = 32
    elif n_cols <= 64:
        max_n = 64
    elif n_cols <= 128:
        max_n = 128
    elif n_cols <= 256:
        max_n = 256
    elif n_cols <= 512:
        max_n = 512
    else:
        max_n = 1024

    _prefix_kernel[(batch_size,)](
        in_0,
        in_1,
        tmp2,
        batch_size,
        n_cols,
        in_0.stride(0),
        in_1.stride(0),
        tmp2.stride(0),
        MAX_N=max_n,
        num_warps=1,
        num_stages=1,
    )

    tmp7 = tmp2.unsqueeze(0).expand(3, -1, -1)
    return tmp7


def replacement_func():
    return fused_prefix