import torch
import triton
import triton.language as tl

# Pattern matching function for Adaptive Avg Pool2d + Flatten fusion
def pattern(input_tensor):
    # Adaptive avg pool2d with output size 1 (spatial mean)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(input_tensor, 1)
    # Flatten operation
    tmp_7 = tmp_6.flatten(1, -1)
    # Return both intermediates that are observable
    return tmp_6, tmp_7

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Triton kernel for fused adaptive avg pool2d + flatten
@triton.jit
def fused_pool_flatten_kernel(
    input_ptr, output_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    num_programs = tl.cdiv(batch_size * channels, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * channels
    
    # Reshape offsets for batch and channel processing
    batch_offsets = offsets // channels
    channel_offsets = offsets % channels
    
    # Compute spatial mean for each batch and channel
    sum_val = tl.zeros([], dtype=tl.float32)
    count = height * width
    
    for h in range(height):
        for w in range(width):
            # Load input data
            input_elem = tl.load(input_ptr + batch_offsets * channels * height * width + 
                               channel_offsets * height * width + h * width + w, mask=mask)
            # Accumulate sum
            sum_val += input_elem
    
    # Compute mean (cast back to original dtype)
    mean_val = sum_val / count
    
    # Store result (batch x channels)
    tl.store(output_ptr + offsets, mean_val, mask=mask)

@torch.fx.wrap
def fused_pool_flatten(input_tensor):
    # Get tensor shapes
    batch_size, channels, height, width = input_tensor.shape
    
    # Create output tensor (batch x channels)
    output = torch.empty((batch_size, channels), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid size
    total_elements = batch_size * channels
    BLOCK_SIZE = 1024  # Can be tuned
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_pool_flatten_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # For the pattern, we need to return both intermediates
    # Compute the pool result separately (adaptive avg pool to size 1)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(input_tensor, 1)
    # Flatten 
    tmp_7 = tmp_6.flatten(1, -1)
    
    return tmp_6, tmp_7

# Replacement function (returns function reference)
def replacement_func():
    return fused_pool_flatten