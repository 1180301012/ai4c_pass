import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern: Alternative element-wise operations for small tensors
    """
    result = (in_3 + in_2) * in_1 + in_0
    return result

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def small_tensor_kernel(
    bias_ptr,
    weight_ptr,
    tensor1_ptr, 
    tensor2_ptr,
    out_ptr,
    total_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for small tensors with better occupancy"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < total_elements
    
    # Coalesced memory access pattern
    tensor1 = tl.load(tensor1_ptr + offset, mask=mask, other=0.0)
    tensor2 = tl.load(tensor2_ptr + offset, mask=mask, other=0.0)
    
    # Optimized broadcasting using warp-level cooperation
    warp_idx = offset // tl.program_id(1)
    weight_bias_idx = warp_idx % tl.constexpr(512)  # Assume max dim2 size
    
    weight = tl.load(weight_ptr + weight_bias_idx, mask=weight_bias_idx < 512, other=0.0)
    bias = tl.load(bias_ptr + weight_bias_idx, mask=weight_bias_idx < 512, other=0.0)
    
    # Vectorized computation
    result = (tensor1 + tensor2) * weight + bias
    
    # Coalesced store
    tl.store(out_ptr + offset, result, mask=mask)

@torch.fx.wrap
def small_tensor_ops(bias, weight, tensor1, tensor2):
    # Get tensor shapes
    shape = tensor1.shape
    dim0_size, dim1_size, dim2_size = shape
    
    # Total elements for small tensor optimization
    total_elements = dim0_size * dim1_size * dim2_size
    
    # Specialized block size for small tensors
    if total_elements <= 4096:
        BLOCK_SIZE = 128
    else:
        BLOCK_SIZE = 256
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(tensor1)
    
    # Launch kernel with optimized configuration
    small_tensor_kernel[(num_programs,)](
        bias_ptr=bias,
        weight_ptr=weight,
        tensor1_ptr=tensor1,
        tensor2_ptr=tensor2,
        out_ptr=output,
        total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return small_tensor_ops