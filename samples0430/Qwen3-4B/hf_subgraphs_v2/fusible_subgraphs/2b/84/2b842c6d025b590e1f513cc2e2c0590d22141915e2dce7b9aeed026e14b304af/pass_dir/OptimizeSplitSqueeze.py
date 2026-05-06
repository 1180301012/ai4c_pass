import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp1 = in_0 * 1000000.0
    tmp2 = in_1 - tmp1
    split = tmp2.split(1, dim=-1)
    tmp4 = split[0]
    tmp5 = split[1]
    tmp6 = tmp4.squeeze(-1)
    tmp7 = tmp6.contiguous()
    tmp8 = tmp5.squeeze(-1)
    tmp9 = tmp8.contiguous()
    return tmp7, tmp9

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_kernel(
    in0_ptr,
    in1_ptr,
    out0_ptr,
    out1_ptr,
    B: tl.int32,
    seq_len: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (B * seq_len)
    in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    in1_0 = tl.load(in1_ptr + offsets, mask=mask, other=0.0)
    in1_1 = tl.load(in1_ptr + offsets + (B * seq_len), mask=mask, other=0.0)
    scaled_in0 = in0 * 1e6
    out0 = in1_0 - scaled_in0
    out1 = in1_1 - scaled_in0
    tl.store(out0_ptr + offsets, out0, mask=mask)
    tl.store(out1_ptr + offsets, out1, mask=mask)

@torch.fx.wrap
def fused_kernel_wrapper(in0, in1):
    B = in0.shape[0]
    seq_len = in0.shape[1]
    out0 = torch.empty((B, seq_len), dtype=in1.dtype)
    out1 = torch.empty((B, seq_len), dtype=in1.dtype)
    BLOCK_SIZE = 1024
    num_blocks = (B * seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    fused_kernel[(num_blocks,)](
        in0_ptr=in0,
        in1_ptr=in1,
        out0_ptr=out0,
        out1_ptr=out1,
        B=B,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out0, out1
def replacement_func():
    return fused_kernel_wrapper