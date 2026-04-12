import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Simple pattern: Just Conv2D to test matching
    """
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def optimized_conv_kernel(
    input_ptr,     # [N, cin, H, W] - input tensor
    weight_ptr,    # [cout, cin, 1, 1] - convolution weight
    bias_ptr,      # [cout] - bias tensor
    output_ptr,    # [N, cout, H, W] - output tensor  
    n_batch,       # batch size (N)
    n_channels_in, # input channels (cin)
    n_channels_out,# output channels (cout)
    height,        # input height
    width,         # input width
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # More efficient 2D grid launch
    m = tl.program_id(0)  # batch position
    n = tl.program_id(1)  # output channel
    
    # Each warp handles a block of spatial positions
    pid_x = tl.program_id(2) if tl.num_programs(2) > 1 else 0
    start_n = pid_x * BLOCK_SIZE_N
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    spatial_mask = offsets_n < height * width
    
    # Initialize results for all spatial positions in this block
    results = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    
    # Load bias once for all spatial positions
    bias_val = tl.load(bias_ptr + n, mask=(n < n_channels_out), other=0.0)
    results += bias_val
    
    # Process all input channels for 1x1 conv (vectorized over spatial positions)
    for c in range(n_channels_in):
        # Input offsets: batch -> cin -> spatial (vectorized)
        input_offsets = (m * n_channels_in + c) * height * width + offsets_n
        
        # Weight offsets: cout -> cin (scalar for all spatial positions)
        weight_offset = (n * n_channels_in + c) * 1 * 1
        
        # Bounds masks
        input_mask = spatial_mask & (m < n_batch) & (c < n_channels_in)
        weight_mask = (c < n_channels_in)
        
        # Load values (input is vectorized, weight is scalar broadcast)
        input_vals = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        weight_val = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0)
        
        # Vectorized accumulation: results += weight_val * input_vals
        results += weight_val * input_vals
    
    # Store results back to output (vectorized)
    output_offsets = (m * n_channels_out + n) * height * width + offsets_n
    output_mask = spatial_mask & (m < n_batch) & (n < n_channels_out)
    tl.store(output_ptr + output_offsets, results, mask=output_mask)

@torch.fx.wrap
def fused_attention_conv(in_0, in_1, in_2):
    # Get input dimensions
    n_batch, n_channels_in, height, width = in_2.shape
    n_channels_out = in_1.shape[0]  # output channels
    
    # Allocate output tensor for conv2d: [N, cout, H, W]
    conv_output = torch.empty((n_batch, n_channels_out, height, width), 
                             dtype=in_2.dtype, device=in_2.device)
    
    # Set up launch configuration for the optimized 3D grid kernel
    BLOCK_SIZE_M = 1    # Process one batch at a time
    BLOCK_SIZE_N = 64   # Process spatial positions in blocks
    
    # Calculate grid dimensions: [batch_size, output_channels, num_spatial_blocks]
    num_spatial_blocks = (height * width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_size = (n_batch, n_channels_out, num_spatial_blocks)
    
    # Launch the optimized conv kernel
    optimized_conv_kernel[grid_size](
        input_ptr=in_2,
        weight_ptr=in_1,
        bias_ptr=in_0,
        output_ptr=conv_output,
        n_batch=n_batch,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    # Perform the remaining operations using basic PyTorch ops
    # View to [N, 1, seq_len] (reshape conv output to match attention pattern)
    tmp_3 = conv_output.view(conv_output.shape[0], 1, conv_output.shape[2] * conv_output.shape[3])
    
    # Apply unsqueeze to add final dimension (matching original pattern)
    tmp_5 = tmp_3.unsqueeze(-1)
    
    return tmp_5

def replacement_func():
    return fused_attention_conv