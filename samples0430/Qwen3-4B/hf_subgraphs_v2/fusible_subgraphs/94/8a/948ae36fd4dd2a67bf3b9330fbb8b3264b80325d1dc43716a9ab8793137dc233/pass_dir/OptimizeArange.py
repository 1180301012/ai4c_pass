import torch
import triton
import triton.language as tl

def pattern(in_0):
    seq = torch.arange(0, 512, device='cuda:0')
    bool_tensor = in_0.to(device='cuda:0', dtype=torch.bool)
    return (seq, bool_tensor)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_arange_kernel(out_ptr, size, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, size)
    offsets = tl.arange(0, block_end - block_start, device='cuda')
    values = offsets + block_start
    tl.store(out_ptr + block_start, values, mask=offsets < (block_end - block_start))

@torch.fx.wrap
def kernel_wrapper(in_0):
    size = 512
    out = torch.empty(size, device='cuda:0', dtype=torch.int32)
    num_blocks = (size + 256 - 1) // 256
    optimized_arange_kernel[(num_blocks,)](\n        out_ptr=out,\n        size=size,\n        BLOCK_SIZE=256,\n    )
    bool_tensor = in_0.to(device='cuda:0', dtype=torch.bool)
    return (out, bool_tensor)

def replacement_func():
    return kernel_wrapper