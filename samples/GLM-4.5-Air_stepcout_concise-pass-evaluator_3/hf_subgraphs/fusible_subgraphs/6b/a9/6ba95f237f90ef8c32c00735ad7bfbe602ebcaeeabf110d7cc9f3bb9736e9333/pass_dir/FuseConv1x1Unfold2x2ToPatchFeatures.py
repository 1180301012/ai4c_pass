import torch
import triton
import triton.language as tl
import math
from torch import Tensor


# Pattern matching function - exactly mirrors the computation in model.py
def pattern(in_0, in_1):
    """Match conv2d + unfold + reshape pattern"""
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=(2, 2), stride=(2, 2))
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return tmp_3


# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_conv1x1_unfold_kernel(
    # Input tensors
    input_ptr,              # in_1: [1, 256, 32, 32]
    weight_ptr,            # in_0: [128, 256, 1, 1]
    # Output tensor  
    output_ptr,            # [1, 128, 4, 256]
    # Metadata
    input_batch: tl.constexpr,
    input_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_channels: tl.constexpr,
    # Stride and kernel size for unfold
    unfold_kernel_h: tl.constexpr,
    unfold_kernel_w: tl.constexpr,
    unfold_stride_h: tl.constexpr,
    unfold_stride_w: tl.constexpr,
    # Block sizes for triton
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    # Each thread block handles a portion of the output channel dimension (batch=1 fixed)
    pid = tl.program_id(0)
    total_output_channels = output_channels
    
    # Map program ID to output channel range
    channel_start = pid * BLOCK_M
    channel_end = min((pid + 1) * BLOCK_M, total_output_channels)
    
    if channel_start >= total_output_channels:
        return
    
    # Compute output spatial dimensions (after unfolding)
    output_patches_h = input_height // unfold_stride_h  # 32//2 = 16
    output_patches_w = input_width // unfold_stride_w    # 32//2 = 16
    total_patches = output_patches_h * output_patches_w  # 16*16 = 256
    
    # Output spatial size after unfolding per channel group
    unfold_output_spatial = unfold_kernel_h * unfold_kernel_w  # 2*2 = 4
    
    # Iterate over the output channels assigned to this block
    for k in range(channel_start, channel_end):
        # Iterate over all output patches
        for patch_idx in range(total_patches):
            # Convert patch index to spatial coordinates
            patch_h = patch_idx // output_patches_w
            patch_w = patch_idx % output_patches_w
            
            # Compute the starting location of this patch in the input
            input_start_h = patch_h * unfold_stride_h
            input_start_w = patch_w * unfold_stride_w
            
            # Initialize output accumulator for this patch and channel
            output_vals = [0.0 for _ in range(unfold_output_spatial)]
            
            # Iterate over input channels to compute the dot product
            for c in range(input_channels):
                # Load the weight for this output channel and input channel
                weight_offset = k * input_channels + c
                weight_val = tl.load(weight_ptr + weight_offset)
                
                # Iterate over the 2x2 patch
                for di in range(unfold_kernel_h):
                    for dj in range(unfold_kernel_w):
                        input_h = input_start_h + di
                        input_w = input_start_w + dj
                        
                        # Bounds check
                        if input_h < input_height and input_w < input_width:
                            # Load input value
                            input_offset = (input_batch * input_channels + c) * input_height * input_width + input_h * input_width + input_w
                            input_val = tl.load(input_ptr + input_offset)
                            
                            # Patch element index:
                            patch_idx_local = di * unfold_kernel_w + dj
                            
                            # Accumulate: conv2d operation on this patch element
                            output_vals[patch_idx_local] += input_val * weight_val
            
            # Store the result for this output channel and patch
            output_offset_base = k * total_patches * unfold_output_spatial
            output_offset_patch = patch_idx * unfold_output_spatial
            
            for elem_idx in range(unfold_output_spatial):
                final_offset = output_offset_base + output_offset_patch + elem_idx
                tl.store(output_ptr + final_offset, output_vals[elem_idx])


@torch.fx.wrap
def fused_conv1x1_unfold(input: Tensor, weight: Tensor) -> Tensor:
    """
    Fused implementation of conv2d(1x1) + unfold(2x2) + reshape
    Directly computes patch features from input using 1x1 convolution weights
    """
    # Input shapes
    input_shape = input.shape  # [1, 256, 32, 32]
    weight_shape = weight.shape  # [128, 256, 1, 1] 
    
    assert input_shape[0] == 1, "Only batch size 1 supported"
    assert input_shape[1] == weight_shape[1], "Input channels must match"
    assert weight_shape[2] == 1 and weight_shape[3] == 1, "Only 1x1 convolution weights supported"
    
    # Output shape should be [1, 128, 4, 256]
    output_channels = weight_shape[0]
    output_patches_h = input_shape[2] // 2  # 32//2 = 16
    output_patches_w = input_shape[3] // 2  # 32//2 = 16
    unfold_output_spatial = 4  # 2x2 patch
    
    output_shape = (1, output_channels, unfold_output_spatial, output_patches_h * output_patches_w)
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Triton kernel launch parameters
    # Use block sizes that work well for the output channel dimension
    BLOCK_M = 32  # Number of output channels per block
    BLOCK_N = 1   # Not used in this kernel
    
    # Number of programs needed for the output channel dimension
    num_programs = (output_channels + BLOCK_M - 1) // BLOCK_M
    
    # Launch the kernel
    fused_conv1x1_unfold_kernel[(num_programs,)](
        input_ptr=input,
        weight_ptr=weight.view(-1),  # Flatten to [128*256]
        output_ptr=output.view(-1),   # Flatten to [1*128*4*256]
        input_batch=input_shape[0],
        input_channels=input_shape[1],
        input_height=input_shape[2],
        input_width=input_shape[3],
        output_channels=output_channels,
        unfold_kernel_h=2,
        unfold_kernel_w=2,
        unfold_stride_h=2,
        unfold_stride_w=2,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )
    
    return output


# Replacement function (returns function reference)
def replacement_func():
    return fused_conv1x1_unfold