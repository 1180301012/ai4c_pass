import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    return (tmp_1, tmp_2, tmp_3, tmp_0)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # SiLU: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_forward(in_0, in_1, in_2):
    # Apply optimized SiLU to in_0
    n_elements = in_0.numel()
    block_size = 1024
    num_programs = (n_elements + block_size - 1) // block_size
    
    # Create output tensor for SiLU result
    tmp_0 = torch.empty_like(in_0)
    
    # Apply optimized SiLU using Triton kernel
    silu_kernel[(num_programs,)](
        x_ptr=in_0,
        out_ptr=tmp_0,
        n_elements=n_elements,
        BLOCK_SIZE=block_size,
    )
    
    # Apply detach operations to maintain the same behavior as original
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    
    return (tmp_1, tmp_2, tmp_3, tmp_0)

def replacement_func():
    return optimized_forward