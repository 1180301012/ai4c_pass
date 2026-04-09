import torch
import triton
import triton.language as tl

# Pattern matching function - matches sigmoid -> view -> multiply sequence
def pattern(in_0, in_1):
    tmp_0 = torch.sigmoid(in_0)
    tmp_1 = tmp_0.view(1, 512, 1, 1)
    tmp_2 = in_1 * tmp_1
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized Triton kernel
@triton.jit
def optimized_sigmoid_view_mul_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input_1 data (the larger tensor [1, 512, 64, 64])
    input_1_val = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate channel index for each offset efficiently
    elements_per_channel = 64 * 64  # height * width
    channel_indices = offsets // elements_per_channel
    
    # Load sigmoid values for each channel from input_0 [1, 512]
    sigmoid_input = tl.load(x_ptr + channel_indices, mask=mask, other=0.0)
    sigmoid_vals = 1.0 / (1.0 + tl.exp(-sigmoid_input.to(tl.float32))).to(input_1_val.dtype)
    
    # Perform fused multiplication with broadcast
    result = input_1_val * sigmoid_vals
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_sigmoid_view_mul(in_0, in_1):
    # Calculate total elements (flatten the 4D tensor to 1D)
    n_elements = in_1.numel()  # 512 * 64 * 64
    
    # Create output tensor with same shape as input_1
    output = torch.empty_like(in_1)
    
    # Use larger block size for better performance
    BLOCK_SIZE = 2048
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_sigmoid_view_mul_kernel[(num_programs,)](
        in_0,
        in_1,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_sigmoid_view_mul