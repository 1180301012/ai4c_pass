import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Match the pattern: hardtanh -> adaptive_avg_pool2d -> view -> flatten
    The operations exactly mirror the model.py computation.
    """
    tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    # Return both intermediate values for proper dataflow matching
    tmp_2 = tmp_1.view(1, -1)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def hardtanh_adaptive_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    num_channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    Fused kernel that applies hardtanh (clamp to [0, 6]) and computes 
    adaptive average pooling to (1, 1) in a single pass.
    
    For each batch and channel, we:
    1. Apply hardtanh (clamp values to [0, 6])
    2. Sum all values across spatial dimensions
    3. Divide by total spatial size (height * width)
    """
    # Get batch and channel indices
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Calculate the number of spatial elements
    spatial_size = height * width
    
    # Initialize sum
    sum_val = 0.0
    
    # Iterate over all spatial positions
    for h in range(height):
        for w in range(width):
            # Calculate the flat index for the input
            idx = ((pid_b * num_channels + pid_c) * height + h) * width + w
            
            # Load the value
            val = tl.load(input_ptr + idx)
            
            # Apply hardtanh: clamp to [0, 6]
            val = tl.where(val < 0.0, 0.0, val)
            val = tl.where(val > 6.0, 6.0, val)
            
            # Accumulate sum
            sum_val += val
    
    # Compute average
    avg = sum_val / tl.cast(spatial_size, tl.float32)
    
    # Calculate output index
    out_idx = pid_b * num_channels + pid_c
    
    # Store result
    tl.store(output_ptr + out_idx, avg)


@torch.fx.wrap
def hardtanh_adaptive_avg_pool2d_wrapper(in_0):
    """
    Wrapper function that launches the Triton kernel.
    Handles the fused hardtanh + adaptive_avg_pool2d + view + flatten operations.
    """
    batch_size = in_0.shape[0]
    num_channels = in_0.shape[1]
    height = in_0.shape[2]
    width = in_0.shape[3]
    
    # Output shape: [batch_size, num_channels]
    output = torch.empty((batch_size, num_channels), dtype=torch.float32, device=in_0.device)
    
    # Define block sizes
    # Each program processes one batch and one channel
    grid = (batch_size, num_channels)
    
    # Launch kernel - using small BLOCK_SIZE for flexibility
    hardtanh_adaptive_avg_pool2d_kernel[grid](
        in_0,
        output,
        batch_size,
        num_channels,
        height,
        width,
        BLOCK_SIZE_B=1,
        BLOCK_SIZE_C=1,
    )
    
    return output


def replacement_func():
    return hardtanh_adaptive_avg_pool2d_wrapper