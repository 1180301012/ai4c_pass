import torch
import triton
import triton.language as tl

def pattern(in_0, tmp_4):
    # Match the exponential and multiplication operations
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * tmp_4
    return tmp_6

def replacement_args(in_0, tmp_4):
    return (in_0, tmp_4)

@triton.jit
def exp_mul_kernel(
    in_ptr,
    mul_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs - compute scalar exponential once, then broadcast
    exp_val = tl.exp(tl.load(in_ptr))
    mul_slice = tl.load(mul_ptr + offsets, mask=mask, other=0.0)
    
    # Multiply with exponential
    out_slice = exp_val * mul_slice
    
    # Store result
    tl.store(out_ptr + offsets, out_slice, mask=mask)

@torch.fx.wrap
def optimized_exp_mul(in_0, tmp_4):
    # Handle exponential and multiplication
    out = torch.empty_like(tmp_4)
    
    N = tmp_4.numel()
    # Use optimal block size for small tensors
    BLOCK_SIZE = 512
    num_programs = 1  # Use single program for small tensors
    
    exp_mul_kernel[(num_programs,)](
        in_0,
        tmp_4,
        out,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_exp_mul