import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.softmax(in_2, dim=2)
    tmp_3 = tmp_2.reshape(-1, 17, 64, 64)
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(128, 17, -1)
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    tmp_7 = tmp_3.mul(in_1)
    tmp_8 = tmp_7.reshape(128, 17, -1)
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    return tmp_3, in_0, in_1, tmp_10

def replacement_args(tmp_3, in_0, in_1, tmp_10):
    return (tmp_3, in_0, in_1)

@triton.jit
def fused_multiply_sum_kernel(
    softmax_ptr,
    in0_ptr,
    in1_ptr,
    sum0_partial_ptr,
    sum1_partial_ptr,
    batch,
    n_channels,
    n_h,
    n_w,
    BLOCK_SIZE_W: tl.constexpr = 64,
):
    idx = tl.program_id(0)
    batch_idx = idx // (n_channels * n_h)
    channel_idx = (idx % (n_channels * n_h)) // n_h
    h = idx % n_h

    softmax_offset = batch_idx * n_channels * n_h * n_w + channel_idx * n_h * n_w + h * n_w
    in1_val = tl.load(in1_ptr + h, dtype=tl.float16)

    sum0 = 0.0
    sum1 = 0.0

    for w in range(n_w):
        softmax_val = tl.load(softmax_ptr + softmax_offset + w, dtype=tl.float16)
        in0_val = tl.load(in0_ptr + w, dtype=tl.float16)
        sum0 += softmax_val * in0_val
        sum1 += softmax_val * in1_val

    sum0_partial_offset = batch_idx * n_channels * n_h + channel_idx * n_h + h
    sum1_partial_offset = batch_idx * n_channels * n_h + channel_idx * n_h + h
    tl.store(sum0_partial_ptr + sum0_partial_offset, sum0, mask=True)
    tl.store(sum1_partial_ptr + sum1_partial_offset, sum1, mask=True)

@torch.fx.wrap
def kernel_wrapper(tmp_3, in_0, in_1):
    batch = tmp_3.size(0)
    n_channels = 17
    n_h = 64
    n_w = 64

    sum0_partial = torch.empty((batch, n_channels, n_h), device=tmp_3.device, dtype=tmp_3.dtype)
    sum1_partial = torch.empty((batch, n_channels, n_h), device=tmp_3.device, dtype=tmp_3.dtype)

    grid = (batch * n_channels * n_h, )
    fused_multiply_sum_kernel[grid](
        tmp_3, 
        in_0, 
        in_1, 
        sum0_partial, 
        sum1_partial,
        batch, 
        n_channels, 
        n_h, 
        n_w,
        BLOCK_SIZE_W=64,
    )

    tmp_10 = torch.empty((batch, n_channels, 2), device=tmp_3.device, dtype=tmp_3.dtype)

    grid = (batch * n_channels,)
    fused_multiply_sum_kernel[grid](
        tmp_3,
        in_0,
        in_1,
        tmp_10,
        batch,
        n_channels,
        n_h,
        n_w,
    )
    return tmp_3, tmp_10
    return tmp_3, tmp_10

def replacement_func():
    return kernel_wrapper