import torch
import triton
import triton.language as tl

# Pattern matching function for addition + dropout2d
def pattern(in_4, in_3):
    """Match addition followed by dropout2d with exact parameters from the model"""
    tmp_3 = in_4 + in_3
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    return tmp_4

# Argument extraction function
def replacement_args(in_4, in_3):
    return (in_4, in_3)

# Optimized fused addition + dropout2d kernel
@triton.jit
def fused_add_dropout_kernel(
    x1_ptr,
    x2_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for spatial blocks
    pid = tl.program_id(0)
    spatial_elements = height * width
    
    # Each program handles a block of spatial elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < spatial_elements
    
    # Loop through each channel
    for c in range(channels):
        channel_offset = c * spatial_elements
        
        # Load input tensors for this channel
        x1_base = x1_ptr + channel_offset
        x2_base = x2_ptr + channel_offset
        x1 = tl.load(x1_base + offsets, mask=mask, other=0.0)
        x2 = tl.load(x2_base + offsets, mask=mask, other=0.0)
        
        # Add tensors
        sum_val = x1 + x2
        
        # Apply dropout: scale by 1/(1-p) during training, use identity during inference
        # Since the model uses dropout=False, we just return the sum
        out_val = sum_val
        
        # Store result
        output_base = output_ptr + channel_offset
        tl.store(output_base + offsets, out_val, mask=mask)

@torch.fx.wrap
def fused_add_dropout(x1, x2, p=0.1, training=False):
    """Fused addition and dropout operation"""
    batch_size, channels, height, width = x1.shape
    
    # Create output tensor
    output = torch.empty_like(x1)
    
    # During inference (training=False), dropout is identity operation
    # So we just perform element-wise addition
    if not training:
        # Use optimized kernel for addition
        BLOCK_SIZE = 1024  # Optimal block size for spatial dimensions
        num_blocks = ((height * width) + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Launch kernel
        grid = (num_blocks,)
        fused_add_dropout_kernel[grid](
            x1_ptr=x1,
            x2_ptr=x2,
            output_ptr=output,
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            p=p,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        # During training, we still just do addition since dropout is disabled
        output = x1 + x2
    
    return output

# Replacement function
def replacement_func():
    return fused_add_dropout