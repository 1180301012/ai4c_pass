import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    """
    Pattern matches the entire computation from inputs to outputs
    """
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.pad(conv2d, [2, 2, 2, 2], 'constant', None)
    tmp_3 = tmp_2.unfold(2, 12, 8)
    tmp_4 = tmp_3.unfold(3, 12, 8)
    tmp_5 = tmp_4.reshape(8, 80, 4, -1)
    tmp_6 = tmp_5.permute(0, 2, 3, 1)
    split = torch.functional.split(tmp_6, [16, 64], dim=-1)
    tmp_8 = split[0]
    tmp_9 = split[1]
    tmp_10 = tmp_8.transpose(-1, -2)
    return tmp_10, tmp_9

def replacement_args(in_0, in_1):
    """
    Extract arguments for the replacement kernel
    """
    return (in_0, in_1)



@triton.jit
def fused_conv2d_unfold_reshape_split_kernel(
    in_0_ptr,  # weight tensor [out_channels, in_channels, 1, 1]
    in_1_ptr,  # input tensor [batch, in_channels, height, width]
    tmp_10_out_ptr,
    tmp_9_out_ptr,
    batch,
    in_channels,
    out_channels,
    height,
    width,
    window_size,
    stride,
    split_size_1,
    split_size_2,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    """
    Fused kernel for entire computation: conv2d + padding + unfold + reshape + split + transpose
    """
    # Calculate dimensions after operations
    conv2d_out_h = height
    conv2d_out_w = width
    padded_h = height + 4
    padded_w = width + 4
    h_out = (padded_h - window_size) // stride + 1
    w_out = (padded_w - window_size) // stride + 1
    total_patches = batch * h_out * w_out
    
    # Program ID
    pid = tl.program_id(0)
    n_pids = tl.cdiv(total_patches * window_size * window_size, BLOCK_SIZE_M)
    
    if pid >= n_pids:
        return
    
    # Calculate global position
    patch_idx = pid * BLOCK_SIZE_M // (window_size * window_size)
    elem_idx_in_patch = (pid * BLOCK_SIZE_M) % (window_size * window_size)
    
    if patch_idx >= total_patches:
        return
    
    # Extract patch coordinates from patch_idx
    flat_idx = patch_idx
    p_b = flat_idx // (h_out * w_out)
    flat_idx = flat_idx % (h_out * w_out)
    p_h = flat_idx // w_out
    p_w = flat_idx % w_out
    
    # Element coordinates within the patch
    elem_h = elem_idx_in_patch // window_size
    elem_w = elem_idx_in_patch % window_size
    
    # Calculate effective input coordinates with padding
    # For pad=2, unfold=12, stride=8: input_coord = patch_coord * stride + elem_coord
    # The patch_coord already accounts for the padding offset
    conv2d_h = p_h * stride + elem_h
    conv2d_w = p_w * stride + elem_w  # This accounts for padding
    
    # Check if we're within valid region
    if (conv2d_h < 0 or conv2d_h >= padded_h or 
        conv2d_w < 0 or conv2d_w >= padded_w):
        # Output zero for out-of-bounds access
        return
    
    # Get 1x1 convolution result at this position
    conv_result = 0.0
    
    # For 1x1 convolution: sum over input channels
    for c_in in range(0, in_channels, BLOCK_SIZE_K):
        channels_block = min(BLOCK_SIZE_K, in_channels - c_in)
        
        # Load input and weight
        # in_1 shape: [batch, in_channels, height, width]
        # in_0 shape: [out_channels, in_channels, 1, 1]
        input_offset = (p_b * height + conv2d_h - 2) * width + (conv2d_w - 2) + c_in
        if 0 <= conv2d_h - 2 < height and 0 <= conv2d_w - 2 < width:
            input_val = tl.load(in_1_ptr + input_offset, mask=input_offset < batch * in_channels * height * width, other=0.0)
            weight_ptr = in_0_ptr + c_in * out_channels
            weight_vals = tl.load(weight_ptr, mask=tl.arange(0, channels_block) < out_channels, other=0.0)
            conv_result += input_val * weight_vals
    
    # Now handle the split operation
    # In the original computation, reshape(8, 80, 4, -1) → permute(0, 2, 3, 1) → split([16, 64])
    # This means we need to map to the final output shape structure
    
    # Calculate the split index based on patch position and split sizes
    split_offset = 0
    if elem_h * window_size + elem_w < split_size_1:
        # Goes to tmp_10 (transposed)
        out_offset = ((elem_h * window_size + elem_w) * split_size_1)
        tl.store(tmp_10_out_ptr + out_offset, conv_result)
    elif elem_h * window_size + elem_w < split_size_1 + split_size_2:
        # Goes to tmp_9 (not transposed)
        out_offset = ((elem_h * window_size + elem_w - split_size_1) * split_size_2)
        tl.store(tmp_9_out_ptr + out_offset, conv_result)

@torch.fx.wrap
def fused_conv2d_unfold_reshape_split_func(in_0, in_1):
    """
    Wrapper function for the entire fused computation
    """
    # Get input tensor properties
    in_0 = in_0.contiguous()
    in_1 = in_1.contiguous()
    
    # Get shapes
    batch, in_channels, height, width = in_1.shape
    out_channels, _, _, _ = in_0.shape
    
    # Fixed parameters from the pattern
    window_size = 12
    stride = 8
    split_size_1 = 16
    split_size_2 = 64
    
    # Calculate output shapes from the original pattern
    # tmp_10 shape: [8, 16, -1, 12] → we need to infer the exact shape
    # tmp_9 shape: [8, -1, 12, 64]
    
    # Based on the reshape(8, 80, 4, -1) operation, we can work backwards
    # The computation leads to output shapes that we need to reconstruct
    
    # For simplicity, let's create outputs that can be reshaped to match the expected pattern
    total_patches = batch * ((height + 4 - window_size) // stride + 1) * ((width + 4 - window_size) // stride + 1)
    tmp_10_out = torch.empty((total_patches, window_size, window_size, split_size_1), dtype=in_1.dtype, device=in_1.device)
    tmp_9_out = torch.empty((total_patches, window_size, window_size, split_size_2), dtype=in_1.dtype, device=in_1.device)
    
    # Set block sizes
    BLOCK_SIZE_M = 1024
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_K = 32
    
    # Calculate grid size
    total_elements = total_patches * window_size * window_size
    grid_size = (triton.cdiv(total_elements, BLOCK_SIZE_M),)
    
    # Launch kernel
    fused_conv2d_unfold_reshape_split_kernel[grid_size](
        in_0,
        in_1,
        tmp_10_out,
        tmp_9_out,
        batch,
        in_channels,
        out_channels,
        height,
        width,
        window_size,
        stride,
        split_size_1,
        split_size_2,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K
    )
    
    return tmp_10_out, tmp_9_out

def replacement_func():
    """
    Returns the fused function as a zero-argument function
    """
    return lambda in_0, in_1: fused_conv2d_unfold_reshape_split_func(in_0, in_1)