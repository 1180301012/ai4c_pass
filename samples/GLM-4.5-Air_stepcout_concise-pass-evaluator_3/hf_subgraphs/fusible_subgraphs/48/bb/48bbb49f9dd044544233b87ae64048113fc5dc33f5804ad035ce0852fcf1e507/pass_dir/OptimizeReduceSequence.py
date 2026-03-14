import torch
import triton
import triton.language as tl

def pattern(base_tensor, weight_tensor):
    """Match the multiply-weight -> reshape -> sum pattern"""
    # This matches one of the two computation paths in the model
    result1 = base_tensor.mul(weight_tensor)
    reshaped1 = result1.reshape(1, 17, -1)  # Use fixed batch size for now
    summed1 = torch.sum(reshaped1, dim=2, keepdim=True)
    return summed1

def replacement_args(base_tensor, weight_tensor):
    return (base_tensor, weight_tensor)

@triton.jit
def optimized_reduce_kernel(
    base_ptr,
    weight_ptr,
    output_ptr,
    base_ptr_stride,
    weight_ptr_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Load base tensor elements
    base_vals = tl.load(base_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    
    # Load weight tensor with simple broadcasting
    weight_vals = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE))
    
    # Multiply and reduce
    product = base_vals * weight_vals
    reduced = tl.sum(product)
    
    # Store result
    tl.store(output_ptr + pid, reduced)

@torch.fx.wrap
def optimized_reduce_sequence(base_tensor, weight_tensor):
    """Optimized version of multiply-weight -> reshape -> sum sequence"""
    
    # Get shapes
    base_shape = base_tensor.shape
    weight_shape = weight_tensor.shape
    
    # Compute total elements in base feature dimension
    total_features = base_shape[1] * base_shape[2] * base_shape[3]
    n_kpts = 17
    
    # Prepare output
    output_size = base_shape[0] * n_kpts
    result = torch.empty(output_size, dtype=torch.float32, device=base_tensor.device)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    
    # Adjust grid size based on actual feature dimension
    actual_block_size = min(BLOCK_SIZE, total_features)
    grid_size = (base_shape[0] * n_kpts,)
    
    # Simplified kernel call with basic parameters
    optimized_reduce_kernel[grid_size](
        base_tensor,
        weight_tensor,
        result,
        0,  # base_ptr_stride - using simple offset for now
        0,  # weight_ptr_stride - using simple offset for now
        actual_block_size
    )
    
    # Reshape to match original output format [batch, n_kpts, 1, 1]
    result = result.reshape(base_shape[0], n_kpts, 1, 1)
    
    return result

def replacement_func():
    return optimized_reduce_sequence