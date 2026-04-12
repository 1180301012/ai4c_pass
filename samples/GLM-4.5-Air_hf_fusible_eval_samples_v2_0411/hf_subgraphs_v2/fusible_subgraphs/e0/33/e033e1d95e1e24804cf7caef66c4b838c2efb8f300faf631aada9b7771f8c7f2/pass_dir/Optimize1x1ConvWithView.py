import torch
import triton
import triton.language as tl

# Pattern matching function - match just the conv2d operation
def pattern(in_3, in_1, in_0):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

# Argument extraction function
def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@triton.jit
def optimized_1x1_conv_kernel(
    x_ptr, 
    weight_ptr, 
    bias_ptr, 
    out_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Get program ID and calculate grid position
    # Each program handles a block of output channels (BLOCK_SIZE_N)
    # and a block of spatial positions (BLOCK_SIZE_M)
    pid_m = tl.program_id(0)  # spatial position block
    pid_n = tl.program_id(1)  # output channel block
    
    # Compute offsets for this program
    spatial_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    channel_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create masks - ensure we don't go out of bounds
    spatial_mask = spatial_offset < batch_size * height * width
    channel_mask = channel_offset < out_channels
    
    # Initialize accumulator
    # We'll accumulate in float32 for precision and convert back at the end
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Load bias for the current output channel block
    bias = tl.load(bias_ptr + channel_offset, mask=channel_mask, other=0.0)
    
    # Matrix multiplication loop over input channels
    # Use tl.arange(0, BLOCK_SIZE_K) which is constexpr and mask invalid accesses
    k_range = tl.arange(0, BLOCK_SIZE_K)
    
    for c in range(0, in_channels, BLOCK_SIZE_K):
        # Calculate current valid range
        k_valid = k_range + c < in_channels
        
        # Load input data for input channels in this block
        # x_ptr indexing: [batch, channel, h, w] -> flattened
        x_ptrs = x_ptr + (
            spatial_offset[:, None] * in_channels +  # spatial position
            (k_range[None, :] + c)                  # input channel
        )
        x_mask = spatial_offset[:, None] < batch_size * height * width
        x_mask = x_mask & k_valid[None, :]
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load weight data for these input channels  
        # weight_ptr indexing: [out_channel, in_channel, 1, 1] -> flattened
        weight_ptrs = weight_ptr + (
            (channel_offset[:, None] * in_channels +   # output channel
             (k_range[None, :] + c))                  # input channel  
        )
        weight_mask = channel_offset[:, None] < out_channels
        weight_mask = weight_mask & k_valid[None, :]
        weight = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Compute matrix multiplication with valid masking
        # x has shape [spatial_positions, input_channels]
        # weight has shape [output_channels, input_channels] 
        # We need to transpose weight to get [input_channels, output_channels]
        contrib = tl.dot(x.to(tl.float32), weight.to(tl.float32).T)
        accumulator += contrib
    
    # Add bias (broadcast bias to all spatial positions)
    accumulator = accumulator + bias[None, :]
    
    # Store results
    out_ptrs = out_ptr + (
        spatial_offset[:, None] * out_channels +  # spatial position
        channel_offset[None, :]                 # output channel
    )
    
    # Convert back to original precision (should match input dtype)
    # For now use float16 as most cases use float16/bfloat16
    result = accumulator.to(tl.float16)
    tl.store(out_ptrs, result, mask=spatial_mask[:, None] & channel_mask[None, :])



@torch.fx.wrap
def optimized_conv2d(x, weight, bias):
    # Get input dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels = weight.shape[0]
    
    # Total elements for matrix multiplication
    total_elements = batch_size * height * width
    
    # Set block sizes to powers of 2 for Triton compatibility
    BLOCK_SIZE_M = 64 if total_elements >= 512 else 32      # Spatial positions, use power of 2
    BLOCK_SIZE_N = 64 if out_channels >= 64 else 32         # Output channels, use power of 2  
    BLOCK_SIZE_K = 32 if in_channels >= 32 else 16          # Input channels, use power of 2
    
    # Calculate grid size
    num_blocks_m = (total_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor
    out = torch.empty((batch_size, out_channels, height, width), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    if total_elements > 0 and out_channels > 0:
        optimized_1x1_conv_kernel[(num_blocks_m, num_blocks_n)](
            x_ptr=x,
            weight_ptr=weight,
            bias_ptr=bias,
            out_ptr=out,
            batch_size=batch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            height=height,
            width=width,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return optimized_conv2d