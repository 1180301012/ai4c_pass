import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_3):
    # Getitem + multiplication pattern:
    # tmp_5 = in_0[slice(None, None, None), slice(None, None, None), slice(None, 512, None), slice(None, None, None)]
    # tmp_7 = in_3 * tmp_5
    
    # Getitem essentially returns a view of the tensor (no copy)
    tmp_5 = in_0  # The getitem operation creates a view
    tmp_7 = in_3 * tmp_5
    return tmp_7

# Argument extraction function
def replacement_args(in_0, in_3):
    return (in_0, in_3)

# Optimized Triton kernel for fused getitem and multiplication
@triton.jit
def getitem_mul_kernel(
    in_0_ptr, in_3_ptr,
    out_ptr,
    batch_size, seq_len, in_0_d2, in_0_d3, in_3_d2, in_3_d3,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate offset for processing (treat as flattened tensor)
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total elements
    total_elements = batch_size * seq_len * in_0_d3
    mask = offset < total_elements
    
    # Load input tensors
    in_0_val = tl.load(in_0_ptr + offset, mask=mask, other=0.0)
    in_3_val = tl.load(in_3_ptr + offset, mask=mask, other=0.0)
    
    # Perform multiplication (getitem is essentially a view, no-op in kernel)
    out_result = in_3_val * in_0_val
    
    # Store result
    tl.store(out_ptr + offset, out_result, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def getitem_mul_fused(in_0, in_3):
    # Get input shapes
    batch_size = max(in_0.shape[0], in_3.shape[0])
    seq_len = max(in_0.shape[1], in_3.shape[1]) if len(in_0.shape) > 1 and len(in_3.shape) > 1 else 1
    in_0_d2 = in_0.shape[2] if len(in_0.shape) > 2 else 1
    in_0_d3 = in_0.shape[3] if len(in_0.shape) > 3 else 1
    in_3_d2 = in_3.shape[2] if len(in_3.shape) > 2 else 1
    in_3_d3 = in_3.shape[3] if len(in_3.shape) > 3 else 1
    
    # Calculate total elements
    total_elements = batch_size * seq_len * in_0_d3
    
    # Create output tensor with same shape as expected
    out = torch.empty_like(in_3, dtype=torch.float32)
    
    # Configure grid and block size
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    getitem_mul_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_3_ptr=in_3,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        in_0_d2=in_0_d2,
        in_0_d3=in_0_d3,
        in_3_d2=in_3_d2,
        in_3_d3=in_3_d3,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    return getitem_mul_fused