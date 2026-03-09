import torch
import triton
import triton.language as tl
import math

def pattern(tmp_4, in_5, in_6):
    tmp_5 = tmp_4 * in_6
    tmp_6 = in_5 + tmp_5
    return tmp_6

def replacement_args(tmp_4, in_5, in_6):
    return (tmp_4, in_5, in_6)

@triton.jit
def fused_muladd_kernel(
    tmp_4_ptr,
    in_5_ptr,
    in_6_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all inputs efficiently in one go
    tmp_4 = tl.load(tmp_4_ptr + offsets, mask=mask, other=0.0)
    in_5 = tl.load(in_5_ptr + offsets, mask=mask, other=0.0)
    in_6 = tl.load(in_6_ptr + offsets, mask=mask, other=0.0)
    
    # Fuse operations: tmp_6 = in_5 + (tmp_4 * in_6)
    # Direct computation without intermediate variables for better performance
    result = in_5 + (tmp_4 * in_6)
    
    # Store result directly
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_muladd_kernel_wrapper(tmp_4, in_5, in_6):
    # Ensure inputs are on the same device and have compatible shapes
    assert tmp_4.shape == in_5.shape, "tmp_4 and in_5 must have the same shape"
    
    # Create output tensor
    out = torch.empty_like(tmp_4)
    
    # Use a larger block size for better GPU utilization
    BLOCK_SIZE = 2048
    
    N = tmp_4.numel()
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Ensure in_6 is on the same device and handle broadcasting via PyTorch
    in_6 = in_6.to(tmp_4.device)
    
    # Launch kernel with proper grid configuration
    fused_muladd_kernel[(num_programs,)](
        tmp_4_ptr=tmp_4,
        in_5_ptr=in_5,
        in_6_ptr=in_6,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_muladd_kernel_wrapper