import torch
import triton
import triton.language as tl

def pattern(in_1, in_0):
    """Match Conv2D followed by Sigmoid operation"""
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.sigmoid(conv2d)
    return tmp_2

def replacement_args(in_1, in_0):
    """Extract input tensors for the replacement"""
    return (in_1, in_0)

@triton.jit
def fused_conv_sigmoid_kernel(x_ptr, weight_ptr, out_ptr, batch_size, in_channels, out_channels, height, width):
    """Fused Conv2D + Sigmoid kernel for 1x1 pointwise operations"""
    # Each program handles one output element 
    pid = tl.program_id(0)
    
    # Calculate coordinates from linear index
    batch_idx = pid // (out_channels * height * width)
    if batch_idx >= batch_size:
        return
        
    rem_idx = pid % (out_channels * height * width)
    out_channel_idx = rem_idx // (height * width)
    if out_channel_idx >= out_channels:
        return
        
    spatial_idx = rem_idx % (height * width)
    h_idx = spatial_idx // width
    if h_idx >= height:
        return
        
    w_idx = spatial_idx % width
    if w_idx >= width:
        return
    
    # Initialize accumulator
    acc = 0.0
    
    # Perform convolution (1x1 convolution is just dot product)
    for in_channel_idx in range(in_channels):
        # Load input value
        x_offset = (batch_idx, in_channel_idx, h_idx, w_idx)
        x_val = tl.load(x_ptr + x_offset, mask=True, other=0.0)
        
        # Load weight value  
        w_offset = (out_channel_idx, in_channel_idx, 0, 0)
        w_val = tl.load(weight_ptr + w_offset, mask=True, other=0.0)
        
        # Accumulate dot product
        acc += x_val * w_val
    
    # Apply sigmoid
    sigmoid_out = tl.sigmoid(tl.cast(acc, tl.float32))
    
    # Store result
    out_offset = (batch_idx, out_channel_idx, h_idx, w_idx)
    tl.store(out_ptr + out_offset, sigmoid_out)

@torch.fx.wrap
def fused_conv_sigmoid(x, weight):
    """Wrapper for fused Conv2D + Sigmoid operation"""
    batch_size = x.shape[0]
    in_channels = x.shape[1]
    in_height = x.shape[2]
    in_width = x.shape[3]
    out_channels = weight.shape[0]
    
    out = torch.empty((batch_size, out_channels, in_height, in_width), dtype=x.dtype, device=x.device)
    
    # Launch kernel - one program per output element
    total_elements = batch_size * out_channels * in_height * in_width
    
    fused_conv_sigmoid_kernel[(total_elements,)](
        x_ptr=x,
        weight_ptr=weight,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=in_height,
        width=in_width
    )
    
    return out

def replacement_func():
    """Return the fused conv2d + sigmoid function"""
    return fused_conv_sigmoid