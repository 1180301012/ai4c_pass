import torch
import triton
import triton.language as tl

# Very simple pattern matching - just bmm
def pattern(tmp_1, in_1):
    # Match just the simple bmm operation
    tmp_2 = torch.bmm(tmp_1, in_1)
    return tmp_2

# Argument extraction function  
def replacement_args(tmp_1, in_1):
    return (tmp_1, in_1)

# Simple Triton kernel for batch matrix multiplication [batch, 1, 1] @ [batch, 1, value_dim] 
@triton.jit
def simple_bmm_kernel(
    a_ptr,
    b_ptr, 
    out_ptr,
    batch_size: tl.constexpr,
    value_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Process elements in 1D blocks
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < batch_size * value_dim
    
    # Load batch index and value index  
    batch_idx = offset // value_dim
    value_idx = offset % value_dim
    
    # Load attention weights [batch_size, 1, 1] -> take the first (and only) element
    a = tl.load(a_ptr + batch_idx, 
               mask=batch_idx < batch_size, 
               other=0.0).to(tl.float32)
    
    # Load value states [batch_size, 1, value_dim] 
    b = tl.load(b_ptr + batch_idx * value_dim + value_idx, 
               mask=mask, 
               other=0.0).to(tl.float32)
    
    # Batch matrix multiplication: [batch, 1, 1] @ [batch, 1, value_dim] = [batch, 1, value_dim]
    # Since we have [1,1] @ [1, value_dim], this simplifies to scalar multiplication
    result = a * b
    
    # Store result
    tl.store(out_ptr + offset, result, mask=mask)

@torch.fx.wrap
def simple_bmm_forward(tmp_1, in_1):
    batch_size = tmp_1.shape[0]
    value_dim = in_1.shape[-1]
    
    # Output shape: [batch_size, 1, value_dim]
    output_shape = (batch_size, 1, value_dim)
    output = torch.zeros(output_shape, dtype=tmp_1.dtype, device=tmp_1.device)
    
    # Triton kernel launch configuration
    BLOCK_SIZE = 256  # Moderate block size for good GPU utilization
    total_elements = batch_size * value_dim
    grid = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_bmm_kernel[(grid,)](
        tmp_1,
        in_1,
        output,
        batch_size, value_dim, BLOCK_SIZE
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return simple_bmm_forward