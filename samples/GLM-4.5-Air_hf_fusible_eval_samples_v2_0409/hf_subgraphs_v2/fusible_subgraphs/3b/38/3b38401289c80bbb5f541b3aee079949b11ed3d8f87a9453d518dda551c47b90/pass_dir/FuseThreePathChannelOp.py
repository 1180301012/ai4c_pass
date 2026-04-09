import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_three_path_optimized_kernel(
    # Input tensors
    in_0_ptr,  # [batch, channels, height, width]
    in_1_ptr,  # [batch, 1, height, width]
    # Output tensor
    out_final_ptr,  # [batch, 3, height, width]
    # Tensor metadata
    batch_size,
    channels,
    height,
    width,
    # Scalar constants
    path1_scale: tl.constexpr,
    path1_bias: tl.constexpr,
    path2_scale: tl.constexpr,
    path2_bias: tl.constexpr,
    path3_scale: tl.constexpr,
    path3_bias: tl.constexpr,
):
    # Program ID determines which element we're processing
    pid = tl.program_id(0)
    
    # Each program processes one element: batch * height * width
    batch_idx = pid // (height * width)
    hw_idx = pid % (height * width)
    
    # Check bounds
    if batch_idx >= batch_size:
        return
    
    # Calculate memory offsets
    in_0_batch_offset = batch_idx * height * width
    in_1_batch_offset = batch_idx * height * width
    
    # Load inputs
    # Path 1: in_1 element
    in_1_val = tl.load(in_1_ptr + in_1_batch_offset + hw_idx)
    path1_out = in_1_val * path1_scale + path1_bias
    
    # Paths 2 & 3: in_0[:, 1] and in_0[:, 2] elements
    # channel1 is at channels dimension index 1, channel2 at index 2
    # For in_0 shape [batch, channels, height, width], channel data is:
    # channel1 at [batch_idx, 1, hw_idx//width, hw_idx%width]
    # channel2 at [batch_idx, 2, hw_idx//width, hw_idx%width]
    channel2_offset = batch_idx * channels * height * width + 1 * height * width + hw_idx
    channel3_offset = batch_idx * channels * height * width + 2 * height * width + hw_idx
    
    channel2_val = tl.load(in_0_ptr + channel2_offset)
    channel3_val = tl.load(in_0_ptr + channel3_offset)
    
    path2_out = channel2_val * path2_scale + path2_bias
    path3_out = channel3_val * path3_scale + path3_bias
    
    # Store results directly to final output with proper channel layout
    # Output layout: [batch, 3, height, width]
    # So we need to store path1 at batch_offset, path2 at batch_offset + height*width, path3 at batch_offset + 2*height*width
    
    final_offset = batch_idx * 3 * height * width + hw_idx
    
    # Store all three results at once - much more efficient!
    tl.store(out_final_ptr + final_offset, path1_out)
    tl.store(out_final_ptr + final_offset + height * width, path2_out)
    tl.store(out_final_ptr + final_offset + 2 * height * width, path3_out)

@torch.fx.wrap
def fused_three_path_operation(in_0, in_1):
    # Get tensor dimensions
    batch_size, channels, height, width = in_0.shape
    
    # Final output shape: [batch, 3, height, width]
    output_shape = (batch_size, 3, height, width)
    out_final = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Calculate grid size: one thread per element in batch * height * width
    total_elements = batch_size * height * width
    
    # Use Triton's autotune for optimal performance
    # For small tensors, use larger blocks to reduce kernel launch overhead
    if total_elements < 1024:
        num_warps = 4
        num_ctas = 1
    elif total_elements < 8192:
        num_warps = 4
        num_ctas = (total_elements + 1023) // 1024
    else:
        num_warps = 8
        num_ctas = (total_elements + 2047) // 2048
    
    # Launch optimized kernel that does everything in one go
    fused_three_path_optimized_kernel[(total_elements,)](
        in_0,
        in_1,
        out_final,
        batch_size,
        channels,
        height,
        width,
        0.458,    # path1_scale
        -0.030000000000000027,  # path1_bias
        0.448,    # path2_scale
        -0.08799999999999997,  # path2_bias
        0.45,     # path3_scale
        -0.18799999999999994,  # path3_bias
    )
    
    return out_final

def pattern(in_0, in_1):
    """Match the entire three-path computation pattern"""
    # Path 1: in_1 * scale + bias
    tmp_1 = in_1 * 0.458
    tmp_2 = tmp_1 + -0.030000000000000027
    
    # Path 2: in_0[:, 1] -> unsqueeze -> scale + bias
    tmp_3 = in_0[(slice(None, None, None), 1)]
    tmp_4 = torch.unsqueeze(tmp_3, 1)
    tmp_5 = tmp_4 * 0.448
    tmp_6 = tmp_5 + -0.08799999999999997
    
    # Path 3: in_0[:, 2] -> unsqueeze -> scale + bias  
    tmp_7 = in_0[(slice(None, None, None), 2)]
    tmp_8 = torch.unsqueeze(tmp_7, 1)
    tmp_9 = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    
    # Final concatenation
    tmp_11 = torch.cat((tmp_2, tmp_6, tmp_10), 1)
    
    return tmp_11

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    return fused_three_path_operation