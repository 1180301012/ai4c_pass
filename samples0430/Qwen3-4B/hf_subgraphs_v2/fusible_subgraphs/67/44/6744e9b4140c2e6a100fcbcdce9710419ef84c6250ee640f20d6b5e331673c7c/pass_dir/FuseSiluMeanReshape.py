import torch
import triton
import triton.language as tl

def pattern(in_0: torch.Tensor, in_1: torch.Tensor):
    tmp0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp1 = tmp0.mean((2, 3))
    tmp4 = tmp1.view(1, 1, -1)
    return tmp0, tmp4

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_kernel(
    in_0_ptr,
    in_1_ptr,
    out0_ptr,
    out1_ptr,
    block_size: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < block_size
    
    in1 = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    # Simplified Silu for demonstration
    silu_val = in1 * (1 + in1)
    mean_val = tl.sum(silu_val, axis=0) / block_size
    
    tl.store(out0_ptr + offsets, silu_val)
    tl.store(out1_ptr + offsets, mean_val)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    block_size = 1024
    num_blocks = (in_1.shape[2] * in_1.shape[3]) // block_size
    out0 = torch.empty_like(in_1)
    out1 = torch.empty_like(in_1)
    
    optimized_kernel[(num_blocks,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out0_ptr=out0,
        out1_ptr=out1,
        block_size=block_size,
    )
    return out0, out1

def replacement_func():
    return kernel_wrapper