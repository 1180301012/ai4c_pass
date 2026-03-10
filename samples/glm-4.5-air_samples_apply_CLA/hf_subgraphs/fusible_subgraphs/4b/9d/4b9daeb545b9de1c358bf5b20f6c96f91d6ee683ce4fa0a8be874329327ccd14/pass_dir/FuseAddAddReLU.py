import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_relu_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all three input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused: x + y + z with ReLU activation
    result = x + y + z
    # ReLU activation
    result = tl.where(result > 0, result, 0.0)
    
    # Store the result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_add_relu_triton(in_0, in_1, in_3):
    """High-performance fused addition and ReLU operation using Triton"""
    # Get tensor properties
    N = in_0.numel()
    dtype = in_0.dtype
    
    # Use optimal block size for this tensor size
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(in_0, dtype=dtype)
    
    # Launch the kernel
    fused_add_relu_kernel[(num_programs,)](
        in_0,
        in_1,
        in_3,
        out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern to match the complete computation:
    Addition + ReLU -> chunk operations and return specific chunks
    """
    # fused addition and ReLU
    tmp_2 = torch.nn.functional.relu(in_0 + in_1 + in_3, inplace=False)
    
    # chunk operations - both inputs need to be chunked
    tmp_3 = in_2.chunk(2, dim=1)
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    
    tmp_6 = tmp_2.chunk(2, dim=1) 
    tmp_7 = tmp_6[0]
    tmp_8 = tmp_6[1]
    
    # return the specific pattern of chunks
    return (tmp_4, tmp_7, tmp_5, tmp_8)

def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments needed for the replacement operation"""
    return (in_0, in_1, in_3)

def replacement_func():
    """Return the fused function"""
    return fused_add_relu_triton