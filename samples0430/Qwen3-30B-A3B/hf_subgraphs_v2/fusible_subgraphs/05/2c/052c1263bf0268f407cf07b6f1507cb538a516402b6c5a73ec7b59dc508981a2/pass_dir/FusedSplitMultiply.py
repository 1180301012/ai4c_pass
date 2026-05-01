import torch
import triton
import triton.language as tl

def pattern(in_2, in_4):
    tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = in_2[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * in_4
    return tmp_5

def replacement_args(in_2, in_4):
    return (in_2, in_4)

@triton.jit
def fused_op_kernel(
    in2_ptr,
    in4_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    h = offsets // 256
    j = offsets % 256
    original_pos = tl.where(j < 128, 128 + j, j - 128)
    idx_in2 = h * 256 + original_pos
    in2 = tl.load(in2_ptr + idx_in2, mask=mask, other=0.0)
    in4 = tl.load(in4_ptr + offsets, mask=mask, other=0.0)
    factor = tl.where(j < 128, -1.0, 1.0)
    out = factor * in2 * in4
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in2, in4):
    N = in2.numel()
    BLOCK_SIZE = 1024
    num_blocks = 1
    out = torch.empty_like(in2)
    fused_op_kernel[(num_blocks,)](in2, in4, out, N, BLOCK_SIZE)
    return out

def replacement_func():
    return kernel_wrapper