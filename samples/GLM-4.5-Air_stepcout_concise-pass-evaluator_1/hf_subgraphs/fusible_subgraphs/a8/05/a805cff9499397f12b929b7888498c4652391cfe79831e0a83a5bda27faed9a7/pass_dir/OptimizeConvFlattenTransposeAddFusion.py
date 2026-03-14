import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor, pos_embed):
    # Match the core computation: conv2d -> flatten -> transpose -> add
    tmp_5 = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (16, 16), (0, 0), (1, 1), 1)
    tmp_6 = tmp_5.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = tmp_7 + pos_embed
    # Return all observable values that match the original pattern
    return tmp_7, tmp_8

def replacement_args(input_tensor, weight_tensor, bias_tensor, pos_embed):
    return (input_tensor, weight_tensor, bias_tensor, pos_embed)

@triton.jit
def conv_flatten_transpose_add_kernel(
    input_ptr, weight_ptr, bias_ptr, pos_embed_ptr, output_ptr,
    batch_size, num_channels, height, width, kh, kw,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """Optimized kernel that fuses conv2d, flatten, transpose, and add operations"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate offsets for output patch features
    h_patches = height // 16  # 224 // 16 = 14
    w_patches = width // 16   # 224 // 16 = 14
    num_patches = h_patches * w_patches  # 14 * 14 = 196
    
    # Calculate which patch and channel we're processing
    patch_idx = pid_m
    channel_idx = pid_n
    
    # Convert to 2D patch coordinates
    patch_h = patch_idx // h_patches
    patch_w = patch_idx % h_patches
    
    # Output coordinates after transpose: [batch, patch, channel]
    batch_offset = 0
    patch_offset = patch_idx
    channel_offset = channel_idx
    
    # Load bias
    bias_val = tl.load(bias_ptr + channel_offset)
    
    # Initialize accumulator
    accumulator = bias_val
    
    # Convolution computation
    for kh_idx in range(kh):
        for kw_idx in range(kw):
            # Input coordinates for this kernel
            input_h = patch_h * 16 + kh_idx
            input_w = patch_w * 16 + kw_idx
            
            if input_h < height and input_w < width:
                # Calculate input offset
                input_offset = (batch_offset * 3 + 0) * height * width + input_h * width + input_w
                
                # Weight offset for this channel and kernel position
                weight_offset = channel_idx * 3 * kh * kw + 0 * kh * kw + kh_idx * kw + kw_idx
                
                input_val = tl.load(input_ptr + input_offset)
                weight_val = tl.load(weight_ptr + weight_offset)
                accumulator += input_val * weight_val
    
    # Add positional embedding
    pos_embed_offset = batch_offset * num_patches * num_channels + patch_idx * num_channels + channel_offset
    pos_embed_val = tl.load(pos_embed_ptr + pos_embed_offset)
    result = accumulator + pos_embed_val
    
    # Store result
    output_offset = patch_offset * num_channels + channel_offset
    tl.store(output_ptr + output_offset, result)

@torch.fx.wrap
def conv_flatten_transpose_add_fused(input_tensor, weight_tensor, bias_tensor, pos_embed):
    """Fused operation: conv2d + flatten + transpose + add"""
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels, _, kh, kw = weight_tensor.shape
    
    # Known shapes from metadata
    h_patches = height // 16  # 14
    w_patches = width // 16   # 14
    num_patches = h_patches * w_patches  # 196
    
    output = torch.empty(batch_size, num_patches, out_channels, 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Triton launch configuration
    num_patches_dim = num_patches
    num_channels_dim = out_channels
    
    # Optimized block sizes
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 128
    
    num_patches_programs = (num_patches_dim + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_channels_programs = (num_channels_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    conv_flatten_transpose_add_kernel[
        (num_patches_programs, num_channels_programs)
    ](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor,
        pos_embed_ptr=pos_embed,
        output_ptr=output,
        batch_size=batch_size,
        num_channels=out_channels,
        height=height,
        width=width,
        kh=kh,
        kw=kw,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    return conv_flatten_transpose_add_fused