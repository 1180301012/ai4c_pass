import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire computation sequence
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """
    Pattern: Conv2D → Add → Add (residual) → BatchNorm → Mean
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = in_4
    tmp_5 = in_5
    # Use a placeholder groups value - this will be replaced by the actual value during replacement
    # The key is to match the structure, not the exact value
    tmp_6 = torch.conv2d(in_6, tmp_5, tmp_4, (1, 1), (0, 0), (1, 1), 1)  # groups=1 is a placeholder
    tmp_5 = tmp_4 = None
    tmp_7 = in_7 + tmp_6
    tmp_6 = None
    tmp_8 = tmp_7 + in_6
    tmp_7 = None
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    tmp_8 = tmp_0 = tmp_1 = tmp_3 = tmp_2 = None
    tmp_10 = tmp_9.mean((2, 3), keepdim=True)
    return (tmp_9, tmp_10)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7)

# Optimized fused kernel using Triton
@triton.jit
def fused_conv_add_bn_mean_kernel(
    # Inputs
    in_6_ptr,  # Main input tensor [B, C, H, W]
    in_7_ptr,  # Residual input tensor [B, C, H, W]
    weight_ptr, # Conv weights [C, 1, 1, 1]
    bias_ptr,   # Conv bias [C]
    # BatchNorm params
    bn_mean_ptr,    # Running mean [C]
    bn_var_ptr,     # Running var [C] 
    bn_weight_ptr,  # BN weight [C]
    bn_bias_ptr,    # BN bias [C]
    # Output
    out_ptr,        # Final output [B, C, H, W]
    # Metadata
    B: tl.constexpr,    # Batch size
    C: tl.constexpr,    # Channels
    H: tl.constexpr,    # Height
    W: tl.constexpr,    # Width
    eps: tl.constexpr,  # BN epsilon (1e-05)
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate base offset for current batch
    base_offset = batch_idx * C * H * W
    
    # Process spatial blocks for current batch/channel combination
    spatial_block_start = tl.program_id(2) * BLOCK_SIZE
    spatial_block_end = min(spatial_block_start + BLOCK_SIZE, H * W)
    
    # Handle case where BLOCK_SIZE is larger than remaining elements
    if spatial_block_start >= H * W:
        return
    
    # Create spatial indices for this block
    spatial_offsets = spatial_block_start + tl.arange(0, spatial_block_end - spatial_block_start)
    
    # Calculate global memory offset for current batch/channel/spatial combination
    mem_offset = base_offset + channel_idx * H * W + spatial_offsets
    
    # Load input tensors
    in_6_val = tl.load(in_6_ptr + mem_offset, other=0.0)
    in_7_val = tl.load(in_7_ptr + mem_offset, other=0.0)
    
    # Load conv parameters (broadcast spatially)
    weight_val = tl.load(weight_ptr + channel_idx * 1 * 1 * 1, other=0.0)
    bias_val = tl.load(bias_ptr + channel_idx, other=0.0)
    
    # Conv2D operation (1x1 convolution is element-wise multiplication + bias)
    conv_out = in_6_val * weight_val + bias_val
    
    # First addition: residual + conv output
    add_out1 = in_7_val + conv_out
    
    # Second addition: residual connection + conv output  
    add_out2 = in_6_val + add_out1
    
    # Load batch normalization parameters
    bn_mean = tl.load(bn_mean_ptr + channel_idx, other=0.0)
    bn_var = tl.load(bn_var_ptr + channel_idx, other=0.0)
    bn_weight = tl.load(bn_weight_ptr + channel_idx, other=1.0)
    bn_bias = tl.load(bn_bias_ptr + channel_idx, other=0.0)
    
    # Batch norm: (x - mean) / sqrt(var + eps) * weight + bias
    bn_out = (add_out2 - bn_mean) * tl.rsqrt(bn_var + eps) * bn_weight + bn_bias
    
    # Store final output
    tl.store(out_ptr + mem_offset, bn_out)

@torch.fx.wrap
def fused_conv_add_bn_mean(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    """Fused kernel wrapper"""
    B, C, H, W = in_6.shape
    
    # Allocate output tensor
    out = torch.empty_like(in_6, dtype=torch.float32)
    
    # Choose optimal block size based on tensor dimensions
    if H * W <= 64:
        BLOCK_SIZE = 32
    elif H * W <= 256:
        BLOCK_SIZE = 64
    else:
        BLOCK_SIZE = 128
    
    # Calculate number of spatial blocks needed
    num_spatial_blocks = (H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Calculate grid dimensions: (batch, channel, spatial_block)
    grid_size = (B, C, num_spatial_blocks)
    
    # Launch kernel
    fused_conv_add_bn_mean_kernel[grid_size](
        in_6_ptr=in_6,
        in_7_ptr=in_7,
        weight_ptr=in_5,
        bias_ptr=in_4,
        bn_mean_ptr=in_0,
        bn_var_ptr=in_1,
        bn_weight_ptr=in_3,
        bn_bias_ptr=in_2,
        out_ptr=out,
        B=B,
        C=C,
        H=H,
        W=W,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Compute mean across spatial dimensions using efficient PyTorch operation
    mean_out = out.mean(dim=(2, 3), keepdim=True)
    
    return out, mean_out

# Replacement function (returns function reference)
def replacement_func():
    return fused_conv_add_bn_mean