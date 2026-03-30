import torch
import triton
import triton.language as tl

def pattern(arg1):
    """
    Match: torch.sum(tmp_5, dim=2, keepdim=True)
    
    This matches the sum reduction operation that appears in both computation paths.
    The function should return the sum results for all paths.
    """
    # Since we're matching just the sum operation, we need to access the graph context
    # For simplicity, we'll do a basic optimization of the sum operation
    result = torch.sum(arg1, dim=2, keepdim=True)
    return result

def replacement_args(arg1):
    # Extract the input tensor for the sum operation
    return (arg1,)

# Optimized Triton kernel for sum reduction along dimension 2
@triton.jit
def optimized_sum_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    feature_size: tl.constexpr,
    reduce_size: tl.constexpr,
    dtype: tl.constexpr,
):
    """Sum reduction along dimension 2 with keepdim=True"""
    
    # Program identifiers: one program per element in the output
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    
    # Check bounds
    if batch_idx >= batch_size or feature_idx >= feature_size:
        return
    
    # Initialize accumulator
    if dtype == tl.bfloat16:
        sum_val = tl.zeros([], dtype=tl.float32)  # Use float32 for precision, cast at end
    elif dtype == tl.float16:
        sum_val = tl.zeros([], dtype=tl.float32)  # Use float32 for precision, cast at end  
    else:
        sum_val = tl.zeros([], dtype=tl.float32)
    
    # Sum over the reduce dimension (dimension 2) - vectorized for better performance
    if reduce_size >= 4:  # Use vectorized loads for larger reductions
        # Process 4 elements at a time
        for reduce_idx in range(0, reduce_size, 4):
            # Load 4 consecutive elements
            offsets = (batch_idx * (feature_size * reduce_size) + feature_idx * reduce_size + 
                      reduce_idx + tl.arange(0, 4))
            mask = reduce_idx + tl.arange(0, 4) < reduce_size
            
            input_vals = tl.load(input_ptr + offsets, mask=mask)
            sum_val += tl.sum(input_vals)
    else:
        # Fallback for small reductions
        for reduce_idx in range(reduce_size):
            offset = (batch_idx * feature_size + feature_idx) * reduce_size + reduce_idx
            input_val = tl.load(input_ptr + offset)
            sum_val += input_val
    
    # Cast back to original dtype if needed
    if dtype == tl.bfloat16:
        sum_val = sum_val.to(tl.bfloat16)
    elif dtype == tl.float16:
        sum_val = sum_val.to(tl.float16)
    
    # Store output - handle the 3D layout properly
    # Output shape is [batch_size, feature_size, 1]
    output_offset = batch_idx * feature_size * 1 + feature_idx * 1 + 0
    tl.store(output_ptr + output_offset, sum_val)

@torch.fx.wrap  
def optimized_sum(input_tensor):
    """Optimized sum operation using Triton - sums along dimension 2 with keepdim=True"""
    
    # Get input tensor dimensions
    batch_size = input_tensor.shape[0]
    feature_size = input_tensor.shape[1] 
    reduce_size = input_tensor.shape[2]
    
    # Create output tensor with keepdim=True: [batch_size, feature_size, 1]
    output = torch.zeros((batch_size, feature_size, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Map dtype to Triton dtype
    if input_tensor.dtype == torch.bfloat16:
        triton_dtype = tl.bfloat16
    elif input_tensor.dtype == torch.float16:
        triton_dtype = tl.float16
    else:
        triton_dtype = tl.float32
    
    # Launch kernel with 2D grid: one program per output element
    grid_size = (batch_size, feature_size)
    
    optimized_sum_kernel[grid_size](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        feature_size=feature_size, 
        reduce_size=reduce_size,
        dtype=triton_dtype,
    )
    
    return output

def replacement_func():
    return optimized_sum