import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    silu = torch.nn.functional.silu(conv2d, inplace=False)
    return silu

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Triton kernel for fused Conv2D + SiLU
@triton.jit
def conv_silu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    in_channels,
    out_channels,
    H,
    W,
    BLOCK_SIZE: tl.constexpr = 1024
):
    # Each block processes one out_channel
    out_c = tl.program_id(0)
    # Each thread processes a (h, w) position within the spatial dimensions
    tid = tl.thread_id(0)
    h = tid // W
    w = tid % W
    
    # Calculate input tensor access offset for (h, w) across all channels
    input_offset = h * W + w  # For the batch=0, channel=0-127
    # Load input for this (h, w): in_channels elements
    input_data = tl.load(
        input_ptr + input_offset * in_channels, 
        mask=tl.arange(0, in_channels) < in_channels, 
        other=0.0
    )
    
    # Load weight for this out_c: in_channels elements
    weight_offset = out_c * in_channels
    weight_data = tl.load(
        weight_ptr + weight_offset, 
        mask=tl.arange(0, in_channels) < in_channels, 
        other=0.0
    )
    
    # Compute dot product: sum(input_data * weight_data)
    dot_product = tl.dot(input_data, weight_data)
    # Load bias
    bias = tl.load(bias_ptr + out_c)
    conv_val = dot_product + bias
    
    # Apply SiLU: x * sigmoid(x)
    sigmoid_val = tl.sigmoid(conv_val)
    silu_val = conv_val * sigmoid_val
    
    # Store the result to output
    output_offset = out_c * H * W + h * W + w
    tl.store(output_ptr + output_offset, silu_val)

# Kernel wrapper with torch.fx.wrap
@torch.fx.wrap
def fused_conv_silu(in_0, in_1, in_2):
    # Extract input dimensions
    batch = in_2.shape[0]
    in_channels = in_2.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    out_channels = in_1.shape[0]
    
    # Create output tensor with correct shape: [batch, out_channels, H, W]
    output = torch.empty((batch, out_channels, H, W), dtype=in_2.dtype, device=in_2.device)
    
    # Calculate grid dimensions
    num_blocks = out_channels
    BLOCK_SIZE = H * W  # Each block processes one out_channel across all (H, W)
    
    # Launch the Triton kernel
    conv_silu_kernel[(num_blocks, 1, 1)](
        in_2,
        in_1,
        in_0,
        output,
        in_channels,
        out_channels,
        H,
        W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_conv_silu