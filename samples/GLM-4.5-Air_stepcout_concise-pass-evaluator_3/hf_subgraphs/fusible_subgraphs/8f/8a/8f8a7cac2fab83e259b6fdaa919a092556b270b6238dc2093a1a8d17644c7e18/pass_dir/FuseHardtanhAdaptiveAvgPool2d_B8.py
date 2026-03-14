import torch
import triton
import triton.language as tl

# Pattern matching function - matches hardtanh + adaptive_avg_pool2d + view + flatten
# with batch size 8
def pattern(in_0):
    tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = tmp_1.view(8, -1)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel that fuses hardtanh + adaptive_avg_pool2d + reshape
@triton.jit
def fused_hardtanh_adaptive_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    num_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel in one batch
    # Grid: (batch_size * num_channels,)
    prog_id = tl.program_id(0)
    batch_idx = prog_id // num_channels
    channel_idx = prog_id % num_channels
    
    # Calculate the starting position for this channel
    channel_offset = channel_idx * height * width
    batch_offset = batch_idx * num_channels * height * width
    
    # Initialize sum accumulator
    sum_val = 0.0
    
    # Process all elements in this channel
    for h in range(height):
        for w in range(width):
            offset = batch_offset + channel_offset + h * width + w
            val = tl.load(input_ptr + offset)
            # Apply hardtanh: clip to [0, 6]
            val = tl.where(val < 0.0, 0.0, val)
            val = tl.where(val > 6.0, 6.0, val)
            sum_val += val
    
    # Compute average
    total_elements = height * width
    avg_val = sum_val / total_elements
    
    # Store result at output position
    output_offset = batch_idx * num_channels + channel_idx
    tl.store(output_ptr + output_offset, avg_val)


def fused_hardtanh_adaptive_avg_pool2d(x):
    batch_size = x.shape[0]
    num_channels = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    
    # Output shape: [batch_size, num_channels]
    output = torch.empty((batch_size, num_channels), device=x.device, dtype=x.dtype)
    
    # Launch kernel with grid of batch_size * num_channels programs
    grid = (batch_size * num_channels,)
    
    fused_hardtanh_adaptive_avg_pool2d_kernel[grid](
        x,
        output,
        batch_size,
        num_channels,
        height,
        width,
        BLOCK_SIZE=1,  # Not used in this implementation
    )
    
    return output


# Wrap the function for FX
@torch.fx.wrap
def kernel_wrapper(x):
    return fused_hardtanh_adaptive_avg_pool2d(x)


# Replacement function returns the wrapped kernel
def replacement_func():
    return kernel_wrapper