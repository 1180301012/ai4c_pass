import torch
import triton
import triton.language as tl

def pattern(tmp_11, tmp_14, tmp_10, tmp_13):
    # Element-wise multiplication and addition
    tmp_15 = tmp_11 * tmp_14
    tmp_16 = tmp_10 * tmp_13
    tmp_17 = tmp_15 + tmp_16
    
    # Return the final result
    return tmp_17

def replacement_args(tmp_11, tmp_14, tmp_10, tmp_13):
    return (tmp_11, tmp_14, tmp_10, tmp_13)

@triton.jit
def fused_elementwise_ops_kernel(
    tmp11_ptr, tmp14_ptr, tmp10_ptr, tmp13_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    tmp11 = tl.load(tmp11_ptr + offsets, mask=mask, other=0.0)
    tmp14 = tl.load(tmp14_ptr + offsets, mask=mask, other=0.0)
    tmp10 = tl.load(tmp10_ptr + offsets, mask=mask, other=0.0)
    tmp13 = tl.load(tmp13_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: (tmp_11 * tmp_14) + (tmp_10 * tmp_13)
    term1 = tmp11 * tmp14
    term2 = tmp10 * tmp13
    out = term1 + term2
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_elementwise_ops(tmp_11, tmp_14, tmp_10, tmp_13):
    # Get tensor size
    n_elements = tmp_11.numel()
    
    # Determine block size and grid size
    BLOCK_SIZE = 1024  # Optimal for most GPUs
    num_programs = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate output tensor
    out = torch.empty_like(tmp_11)
    
    # Launch kernel
    fused_elementwise_ops_kernel[(num_programs,)](
        tmp11_ptr=tmp_11,
        tmp14_ptr=tmp_14,
        tmp10_ptr=tmp_10,
        tmp13_ptr=tmp_13,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_elementwise_ops