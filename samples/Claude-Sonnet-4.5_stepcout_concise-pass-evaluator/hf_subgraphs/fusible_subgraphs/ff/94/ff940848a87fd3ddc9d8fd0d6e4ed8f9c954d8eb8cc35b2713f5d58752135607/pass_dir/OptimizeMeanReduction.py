import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Match mean reduction along dim=-2 with keepdim=True"""
    result = input_tensor.mean(dim=-2, keepdim=True)
    return result

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def mean_reduction_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    reduce_dim,
    last_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for mean reduction along dim=-2"""
    # Each program handles one (batch, last_dim) position
    pid = tl.program_id(0)
    batch_idx = pid // last_dim
    last_idx = pid % last_dim
    
    # Compute the mean over the reduce_dim
    sum_val = 0.0
    for i in range(0, reduce_dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < reduce_dim
        
        # Calculate input indices
        indices = batch_idx * reduce_dim * last_dim + offsets * last_dim + last_idx
        
        # Load and accumulate
        vals = tl.load(input_ptr + indices, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
    
    # Compute mean and store
    mean_val = sum_val / reduce_dim
    output_idx = batch_idx * last_dim + last_idx
    tl.store(output_ptr + output_idx, mean_val)

@torch.fx.wrap
def optimized_mean_reduction(input_tensor):
    """Wrapper for optimized mean reduction"""
    batch_size = input_tensor.shape[0]
    reduce_dim = input_tensor.shape[1]
    last_dim = input_tensor.shape[2]
    
    # Output shape: (batch_size, 1, last_dim)
    output = torch.empty((batch_size, 1, last_dim), 
                         dtype=input_tensor.dtype, 
                         device=input_tensor.device)
    
    # Launch kernel
    grid = (batch_size * last_dim,)
    BLOCK_SIZE = 1024
    
    mean_reduction_kernel[grid](
        input_tensor,
        output,
        batch_size,
        reduce_dim,
        last_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_mean_reduction