import torch
import triton
import triton.language as tl

def pattern(conv_out, in_5):
    """Match conv_out -> sigmoid -> multiply sequence"""
    tmp_3 = torch.sigmoid(conv_out)
    tmp_4 = in_5 * tmp_3
    return tmp_4

def replacement_args(conv_out, in_5):
    return (conv_out, in_5)

@triton.jit
def fused_sigmoid_multiply_kernel(conv_out_ptr, in_5_ptr, out_ptr, 
                                  batch_size, channels, in_height, in_width,
                                  BLOCK_SIZE: tl.constexpr):
    """Fused sigmoid + multiply operation with broadcasting"""
    pid = tl.program_id(0)
    n_elements = batch_size * channels * in_height * in_width
    
    # Each program handles a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Convert 1D offset to 4D coordinates for in_5 tensor [batch, channels, height, width]
    offset = offsets
    batch_idx = offset // (channels * in_height * in_width)
    offset = offset % (channels * in_height * in_width)
    
    channel_idx = offset // (in_height * in_width)
    offset = offset % (in_height * in_width)
    
    height_idx = offset // in_width
    width_idx = offset % in_width
    
    # Compute corresponding position in conv_out tensor [batch, channels, 1, 1]
    conv_out_offset = batch_idx * channels + channel_idx
    
    # Load conv_out value (same for all spatial positions in a channel)
    conv_out_val = tl.load(conv_out_ptr + conv_out_offset, 
                          mask=conv_out_offset < (batch_size * channels), 
                          other=0.0)
    
    # Load in_5 value
    in_5_val = tl.load(in_5_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operation: sigmoid then multiply (with broadcasting)
    sigmoid_val = tl.sigmoid(conv_out_val)
    out = sigmoid_val * in_5_val
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_sigmoid_multiply(conv_out, in_5):
    """Wrapper function for launching the fused kernel"""
    # Get shapes from both tensors
    batch_size_conv, channels_conv, conv_height, conv_width = conv_out.shape
    batch_size_in, channels_in, in_height, in_width = in_5.shape
    
    # Validate shapes match for batch and channels
    assert batch_size_conv == batch_size_in, f"Batch size mismatch: {batch_size_conv} vs {batch_size_in}"
    assert channels_conv == channels_in, f"Channel size mismatch: {channels_conv} vs {channels_in}"
    assert conv_height == 1 and conv_width == 1, f"Conv output should be 1x1 spatially, got {conv_height}x{conv_width}"
    
    # Calculate total elements and block size based on in_5 shape (larger tensor)
    n_elements = batch_size_in * channels_in * in_height * in_width
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same shape as in_5
    out = torch.empty_like(in_5)
    
    # Launch kernel
    fused_sigmoid_multiply_kernel[(num_programs,)](
        conv_out_ptr=conv_out,
        in_5_ptr=in_5, 
        out_ptr=out,
        batch_size=batch_size_in,
        channels=channels_in,
        in_height=in_height,
        in_width=in_width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_sigmoid_multiply