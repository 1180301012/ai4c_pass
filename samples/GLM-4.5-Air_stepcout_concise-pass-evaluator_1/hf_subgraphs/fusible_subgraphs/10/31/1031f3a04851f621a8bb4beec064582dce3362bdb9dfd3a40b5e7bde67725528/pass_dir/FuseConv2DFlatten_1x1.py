import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern matching for Conv2D + Flatten operations
    The pattern matches:
    - torch.conv2d with 1x1 kernel, stride=[1,1], padding=[0,0], dilation=[1,1], groups=1
    - Followed by torch.flatten with dim=2
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_1 = tmp_0 = None
    tmp_3 = torch.flatten(tmp_2, 2)
    tmp_2 = None
    return (tmp_3,)

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused Conv2D-Flatten operation"""
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv2d_flatten_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_OUT_CHANNELS: tl.constexpr,
    BLOCK_HEIGHT: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
):
    """Fused Conv2D-Flatten kernel for 1x1 convolution
    
    This kernel performs a 1x1 convolution followed by flattening spatial dimensions.
    For 1x1 convolutions, each output channel at each spatial location is:
    output[b][c][h][w] = sum_{c_in}(input[b][c_in][h][w] * weight[c][c_in]) + bias[c]
    """
    # Get program indices
    batch_idx = tl.program_id(0)
    ch_out_idx = tl.program_id(1)
    h_idx = tl.program_id(2)
    w_idx = tl.program_id(3)
    
    # Handle batch processing with BLOCK_BATCH
    for batch_offset in range(0, batch_size, BLOCK_BATCH):
        actual_batch = min(BLOCK_BATCH, batch_size - batch_offset)
        local_batch_idx = tl.arange(0, actual_batch)
        
        if batch_idx < batch_size:
            # Process one channel at a time for better cache utilization
            ch_out_local = ch_out_idx * BLOCK_OUT_CHANNELS + tl.arange(0, BLOCK_OUT_CHANNELS)
            ch_out_local = ch_out_local[ch_out_local < out_channels]
            
            # Load bias for this output channel
            bias = tl.load(bias_ptr + ch_out_local, other=0.0)
            
            # Compute output for each spatial position
            h_local = h_idx * BLOCK_HEIGHT + tl.arange(0, BLOCK_HEIGHT)
            h_local = h_local[h_local < height]
            w_local = w_idx * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
            w_local = w_local[w_local < width]
            
            # Compute output dimensions for flattening
            spatial_size = height * width
            flattened_idx = (ch_out_local * spatial_size + 
                           h_local[:, None] * width + w_local[None, :])
            
            # Convert to flattened output coordinates
            output_base = (batch_idx + local_batch_idx[:, None, None, None]) * out_channels * spatial_size + flattened_idx[None, :, :, :]
            
            # Compute convolution: sum over input channels
            total = bias[None, :, None, None]  # [1, BLOCK_OUT_CHANNELS, 1, 1]
            
            for c_in in range(0, in_channels, 1):
                # Load weight for this input channel
                weight = tl.load(weight_ptr + ch_out_local[:, None] * in_channels + c_in, other=0.0)
                weight = weight[None, :, None, None]  # [1, BLOCK_OUT_CHANNELS, 1, 1]
                
                # Load input for this spatial position and input channel
                input_base = (batch_idx + local_batch_idx[:, None, None]) * in_channels * height * width + \
                           c_in * height * width + h_local[:, None] * width + w_local
                input_val = tl.load(input_ptr + input_base, other=0.0)
                input_val = input_val[:, None, :, :]  # [BLOCK_BATCH, 1, BLOCK_HEIGHT, BLOCK_WIDTH]
                
                # Accumulate: input * weight
                total += input_val * weight
            
            # Store output
            tl.store(output_ptr + output_base, total)

@torch.fx.wrap
def fused_conv2d_flatten(input, weight, bias):
    """Performs fused Conv2D (1x1) + Flatten operation using Triton
    
    Args:
        input: Input tensor of shape [batch, in_channels, height, width]
        weight: Weight tensor of shape [out_channels, in_channels, 1, 1]
        bias: Bias tensor of shape [out_channels]
    
    Returns:
        Flattened output tensor of shape [batch, out_channels * height * width]
    """
    batch_size, in_channels, height, width = input.shape
    out_channels = weight.shape[0]
    
    # Determine output shape
    output_shape = (batch_size, out_channels * height * width)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Optimize block sizes based on typical GPU architecture
    BLOCK_OUT_CHANNELS = 64  # Number of output channels to process per program
    BLOCK_HEIGHT = 16       # Height tile size
    BLOCK_WIDTH = 16        # Width tile size
    BLOCK_BATCH = 4         # Batch size to process per program
    
    # Calculate grid dimensions
    num_batch = (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH
    num_out_channels = (out_channels + BLOCK_OUT_CHANNELS - 1) // BLOCK_OUT_CHANNELS
    num_height = (height + BLOCK_HEIGHT - 1) // BLOCK_HEIGHT  
    num_width = (width + BLOCK_WIDTH - 1) // BLOCK_WIDTH
    
    # Launch kernel
    fused_conv2d_flatten_kernel[(num_batch, num_out_channels, num_height, num_width)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_OUT_CHANNELS=BLOCK_OUT_CHANNELS,
        BLOCK_HEIGHT=BLOCK_HEIGHT,
        BLOCK_WIDTH=BLOCK_WIDTH,
        BLOCK_BATCH=BLOCK_BATCH,
    )
    
    return output

def replacement_func():
    """Return the fused Conv2D-Flatten function"""
    return fused_conv2d_flatten