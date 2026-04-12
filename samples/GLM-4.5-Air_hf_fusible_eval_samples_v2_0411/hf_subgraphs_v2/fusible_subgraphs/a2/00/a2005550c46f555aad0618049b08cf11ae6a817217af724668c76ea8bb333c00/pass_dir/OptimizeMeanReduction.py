import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    # Mean reduction across spatial dimensions (2, 3) with keepdim=True
    result = input_tensor.mean((2, 3), keepdim=True)
    return result

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def mean_reduction_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    num_features,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one feature in one batch
    pid = tl.program_id(0)
    
    # Calculate which feature and batch this program handles
    feature_idx = pid % num_features
    batch_idx = pid // num_features
    
    # Calculate input tensor offset for this batch and feature
    input_offset = batch_idx * num_features * height * width + feature_idx * height * width
    
    # Initialize accumulator for this spatial slice
    sum_val = tl.zeros([], dtype=tl.float32)
    
    # Iterate over spatial dimensions
    for h in range(height):
        for w in range(width):
            offset = input_offset + h * width + w
            val = tl.load(input_ptr + offset, mask=True)
            sum_val = sum_val + val
    
    # Compute mean: sum / (height * width)
    mean_val = sum_val / (height * width)
    
    # Calculate output offset for this batch and feature
    output_offset = batch_idx * num_features + feature_idx
    
    # Store the result
    tl.store(output_ptr + output_offset, mean_val)

@torch.fx.wrap
def optimized_mean_reduction(input_tensor):
    """
    Optimized mean reduction across spatial dimensions (2, 3) with keepdim=True
    """
    batch_size, num_features, height, width = input_tensor.shape
    
    # Calculate total number of programs (one per feature per batch)
    total_elements = batch_size * num_features
    
    # Determine optimal block size
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with shape [batch_size, num_features, 1, 1]
    output = torch.empty((batch_size, num_features, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel - we need to use a flattened view for the output
    flat_output = output.view(batch_size, num_features)
    
    mean_reduction_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=flat_output,
        batch_size=batch_size,
        num_features=num_features,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_mean_reduction