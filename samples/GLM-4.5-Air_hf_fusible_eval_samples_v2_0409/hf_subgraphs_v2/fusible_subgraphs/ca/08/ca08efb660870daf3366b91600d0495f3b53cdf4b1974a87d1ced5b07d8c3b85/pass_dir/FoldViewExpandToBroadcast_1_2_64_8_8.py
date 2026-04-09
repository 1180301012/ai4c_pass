import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    # This matches: tmp_2 = in_0.view(1, 2, 1, 8, 8)
    # followed by: tmp_3 = tmp_2.expand(1, 2, 64, 8, 8)
    tmp_2 = in_0.view(1, 2, 1, 8, 8)
    tmp_3 = tmp_2.expand(1, 2, 64, 8, 8)
    return tmp_3

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel using Triton - directly broadcast without intermediate view/expand
@triton.jit
def broadcast_kernel(
    input_ptr,
    output_ptr,
    n_batch_in,
    n_channels_in,
    height_in,
    width_in,
    n_channels_out,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program handles a batch/channel slice in output
    pid_m = tl.program_id(0)  # batch
    pid_n = tl.program_id(1)  # output channel
    pid_k = tl.program_id(2)  # channel in block (for vectorization)
    
    # Map output channel to input channel (input only has 2 channels, output has 64)
    input_channel = pid_n % 2  # Cycle through 0, 1 for input channels
    
    # Output stride for [batch, output_channels, height, width] but with 5D layout [1, 2, 64, 8, 8]
    # The effective output shape is [1, 2, 64, 8, 8] but we treat it as [1, 2*64, 8, 8] for simplicity
    output_offset = (pid_m * (2 * n_channels_out) * height_in * width_in + 
                    pid_n * height_in * width_in + 
                    pid_k * BLOCK_SIZE_K)
    
    # Input offset maps to the same spatial position but with input channel
    input_offset = (pid_m * 2 * height_in * width_in + 
                   input_channel * height_in * width_in + 
                   pid_k * BLOCK_SIZE_K)
    
    # Load input data and broadcast to output
    input_val = tl.load(input_ptr + input_offset, 
                       mask=input_offset < (n_batch_in * 2 * height_in * width_in), 
                       other=0.0)
    
    # Store to all positions that would have been repeated in the broadcast
    for k in range(BLOCK_SIZE_K):
        if output_offset + k < (n_batch_in * 2 * n_channels_out * height_in * width_in):
            tl.store(output_ptr + output_offset + k, input_val)

# Kernel wrapper
@torch.fx.wrap
def direct_broadcast_gpu(x):
    # Input shape: [1, 2, 8, 8]
    # Output shape: [1, 2, 64, 8, 8] -> treated as [1, 128, 8, 8] for simplicity
    
    n_batch, n_channels_in, height, width = x.shape
    n_channels_out = 64
    
    # Vectorization size for better memory throughput
    BLOCK_SIZE_K = 32  # Process 32 elements at once (width dimension)
    
    # Calculate grid dimensions
    # Each output channel needs its own program
    grid_m = n_batch  # batch dimension (always 1)
    grid_n = 2 * n_channels_out  # total output channels (2 * 64 = 128)
    grid_k = (width + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # Allocate output tensor with effective 4D shape [batch, n_channels_out*2, height, width]
    # This represents the flattened view of [1, 2, 64, 8, 8]
    out = torch.empty(n_batch, 2 * n_channels_out, height, width, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    broadcast_kernel[(grid_m, grid_n, grid_k)](
        input_ptr=x,
        output_ptr=out,
        n_batch_in=n_batch,
        n_channels_in=n_channels_in,
        height_in=height,
        width_in=width,
        n_channels_out=n_channels_out,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Reshape back to the target 5D shape [1, 2, 64, 8, 8]
    return out.view(1, 2, 64, 8, 8)

# Replacement function (returns function reference)
def replacement_func():
    return direct_broadcast_gpu