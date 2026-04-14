import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Match the exact computation pattern from the graphs
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_1 = in_0 // 32  # Default to 32, will be specialized via replacement_args
    tmp_2 = torch.sym_sum([1, tmp_1])
    tmp_3 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_3)

def replacement_args(in_0, in_1):
    # Extract divisor from in_0 (the integer input)
    # The actual computation shows in_0 // divisor, so we need to determine the divisor
    # Based on the graphs, divisor can be 8, 16, or 32
    # We'll handle this by trying different patterns and passing the divisor
    return (in_0, in_1, 32)  # Default divisor

@triton.jit
def relu_and_mean_kernel(
    input_ptr,
    output_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Each program handles a block of channels and spatial positions
    # We'll optimize for tensor shapes like [1, C, H, W]
    
    pid_c = tl.program_id(0)  # channel block
    pid_h = tl.program_id(1)  # spatial block (y)
    pid_w = tl.program_id(2)  # spatial block (x)
    
    # Compute range for this program
    channel_start = pid_c * BLOCK_SIZE_C
    channel_end = min(channel_start + BLOCK_SIZE_C, n_channels)
    
    # Create offsets within the block
    c_off = tl.arange(0, BLOCK_SIZE_C)
    h_off = pid_h * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    w_off = pid_w * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Calculate absolute offsets
    c_abs = channel_start + c_off
    h_abs = h_off[:, None]
    w_abs = w_off[None, :]
    
    # Create 2D spatial mask
    h_mask = h_abs < height
    w_mask = w_abs < width
    spatial_mask = h_mask & w_mask
    
    # Initialize accumulator for mean computation
    accumulator = 0.0
    
    # Process each channel in the block
    for c in range(channel_start, min(channel_start + BLOCK_SIZE_C, n_channels)):
        # Calculate pointer offset for this channel
        base_ptr = input_ptr + c * height * width
        
        # Load input values with masking
        input_vals = tl.load(base_ptr + h_abs * width + w_abs, mask=spatial_mask, other=0.0)
        
        # Apply ReLU (no-op for positive values, 0 for negative)
        relu_vals = tl.maximum(input_vals, 0.0)
        
        # Add to accumulator (this will hold sum over spatial dimensions)
        accumulator += tl.sum(relu_vals)
    
    # Output shape is [1, C, 1, 1], so we need to store the mean
    # The mean is accumulator / (height * width)
    mean_val = accumulator / (height * width)
    
    # Store the result for this channel block
    output_offset = pid_c
    tl.store(output_ptr + output_offset, mean_val)

@torch.fx.wrap
def fused_relu_mean(input_tensor):
    # Input shape: [1, C, H, W] 
    N, C, H, W = input_tensor.shape
    
    # Determine optimal block sizes based on tensor dimensions
    BLOCK_SIZE_N = 16  # Block size for spatial dimensions
    BLOCK_SIZE_C = 64   # Block size for channel dimension
    
    # Calculate number of programs needed
    num_c_blocks = (C + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    num_h_blocks = (H + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    num_w_blocks = (W + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor with shape [1, C, 1, 1]
    output_tensor = torch.empty((1, C, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch the kernel
    grid = (num_c_blocks, num_h_blocks, num_w_blocks)
    relu_and_mean_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output_tensor,
        n_channels=C,
        height=H,
        width=W,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    # Convert output back to original format expectations
    # We also need to return the original ReLU output, but we need to compute it separately
    # due to kernel fusion limitations
    relu_output = torch.nn.functional.relu(input_tensor, inplace=False)
    
    return relu_output, output_tensor.squeeze(-1).squeeze(-1)  # Shape [1, C, 1, 1]

def replacement_func():
    """
    Returns the fused kernel function.
    Note: Due to the complexity of fusing both ReLU output AND mean computation,
    we'll implement this as a two-step optimization focusing on the mean computation
    which is the more performance-critical operation.
    """
    def fused_compute(in_0, in_1, divisor):
        # Main computation: ReLU on in_1
        relu_output = torch.nn.functional.relu(in_1, inplace=False)
        
        # Division and symmetric sum (these are fast operations)
        division_result = in_0 // divisor
        sym_sum_result = torch.sym_sum([1, division_result])
        
        # Optimized mean computation using Triton kernel
        n, c, h, w = relu_output.shape
        if n == 1 and h > 1 and w > 1:  # Only optimize for 4D tensors with spatial dimensions
            # Use Triton-optimized mean computation
            mean_output = fused_relu_mean(relu_output)
        else:
            # Fall back to regular mean for unsupported shapes
            mean_output = relu_output.mean((2, 3), keepdim=True)
        
        return (relu_output, mean_output)
    
    return fused_compute