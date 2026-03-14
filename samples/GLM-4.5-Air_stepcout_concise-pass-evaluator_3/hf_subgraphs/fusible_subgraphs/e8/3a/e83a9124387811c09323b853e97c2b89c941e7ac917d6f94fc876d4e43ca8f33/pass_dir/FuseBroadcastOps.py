import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern: Simple element-wise operations
    """
    result = (in_3 + in_2) * in_1 + in_0
    return result

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_broadcast_kernel(
    bias_ptr,
    weight_ptr,
    tensor1_ptr, 
    tensor2_ptr,
    out_ptr,
    dim0_size: tl.constexpr,
    dim1_size: tl.constexpr,
    dim2_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized fused kernel for: (tensor2 + tensor1) * weight + bias"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < (dim0_size * dim1_size * dim2_size)
    
    # Load tensor data first (main computation tensors)
    tensor1 = tl.load(tensor1_ptr + offset, mask=mask, other=0.0)
    tensor2 = tl.load(tensor2_ptr + offset, mask=mask, other=0.0)
    
    # Optimized broadcasting: compute indices once
    mod_dim2 = offset % dim2_size
    
    # Optimized weight and bias loading with shared mask
    # Each thread loads its corresponding weight/bias based on position in last dimension
    weight_mask = (mod_dim2 < dim2_size)
    weight = tl.load(weight_ptr + mod_dim2, mask=weight_mask, other=0.0)
    bias = tl.load(bias_ptr + mod_dim2, mask=weight_mask, other=0.0)
    
    # Fused computation with operation reordering for better ILP
    sum_tensor = tensor1 + tensor2
    weighted = sum_tensor * weight
    result = weighted + bias
    
    # Store result
    tl.store(out_ptr + offset, result, mask=mask)

@torch.fx.wrap
def fused_broadcast_ops(bias, weight, tensor1, tensor2):
    # Get tensor shapes
    shape = tensor1.shape
    dim0_size, dim1_size, dim2_size = shape
    
    # Set up grid and block size with dynamic optimization
    total_elements = dim0_size * dim1_size * dim2_size
    
    # Dynamic block size selection based on tensor dimensions
    if total_elements < 8192:
        BLOCK_SIZE = 256  # Smaller blocks for small tensors
    elif total_elements < 65536:
        BLOCK_SIZE = 512  # Medium blocks for medium tensors
    elif total_elements < 262144:
        BLOCK_SIZE = 1024  # Default blocks for large tensors
    else:
        BLOCK_SIZE = 2048  # Larger blocks for very large tensors
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create main output tensor
    output = torch.empty_like(tensor1)
    
    # Launch kernel for main computation
    fused_broadcast_kernel[(num_programs,)](
        bias_ptr=bias,
        weight_ptr=weight,
        tensor1_ptr=tensor1,
        tensor2_ptr=tensor2,
        out_ptr=output,
        dim0_size=dim0_size,
        dim1_size=dim1_size,
        dim2_size=dim2_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_broadcast_ops