import torch
import triton
import triton.language as tl

@triton.jit
def fused_stack_kernel(
    # Input pointers
    in_0_ptr,  # [N, 512, 40, 40]
    in_1_ptr,  # [N, 512, 20, 20] -> needs upsampling to [N, 512, 40, 40]
    in_2_ptr,  # [N, 256, 40, 40] 
    in_3_ptr,  # [N, 256, 40, 40]
    
    # Output pointer
    out_ptr,   # [3, N, 768, 40, 40]
    
    # Tensor shapes
    N: tl.constexpr,
    C: tl.constexpr,
    C2: tl.constexpr,
    
    # Block sizes
    BLOCK_M: tl.constexpr,
):
    # Process each element in the batch
    n = tl.program_id(0)
    c = tl.program_id(1)  # Combined channel index: 0-767
    h = tl.program_id(2)
    w = tl.program_id(3)
    
    # Process channels in blocks
    block_c = tl.arange(0, BLOCK_M)
    
    masks = (block_c < C) & (h < 40) & (w < 40)
    offsets = (n * C + block_c) * (40 * 40) + h * 40 + w
    
    # Load in_0 (direct copy, no interpolation needed)
    in_0_val = tl.load(in_0_ptr + offsets, mask=masks, other=0.0)
    
    # Load in_1 with 2x nearest neighbor upsampling
    in_1_base_offset = n * C * (20 * 20)
    in_1_h_src = (h + 1) // 2  # 2x nearest neighbor upsampling
    in_1_w_src = (w + 1) // 2
    in_1_offsets = in_1_base_offset + (c * C + block_c) * (20 * 20) + in_1_h_src * 20 + in_1_w_src
    in_1_val = tl.load(in_1_ptr + in_1_offsets, mask=masks, other=0.0)
    
    # Load in_2 for concatenation (first half of channels)
    c_in2 = c * C + block_c
    in_2_offsets = (n * C2 + c_in2) * (40 * 40) + h * 40 + w
    in_2_mask = (c_in2 < C2) & (h < 40) & (w < 40)
    in_2_val = tl.load(in_2_ptr + in_2_offsets, mask=in_2_mask, other=0.0)
    
    # Load in_3 for concatenation (second half of channels) 
    c_in3 = c * C + block_c - C2
    in_3_offsets = (n * C2 + c_in3) * (40 * 40) + h * 40 + w
    in_3_mask = (c_in3 < C2) & (c_in3 >= 0) & (h < 40) & (w < 40)
    in_3_val = tl.load(in_3_ptr + in_3_offsets, mask=in_3_mask, other=0.0)
    
    # Create concatenated value for tmp_0
    concat_val = tl.where(c_in2 < C2, in_2_val, 0.0)
    concat_val = tl.where(c_in3 >= 0 & c_in3 < C2, in_3_val, concat_val)
    
    # Determine which stack element this channel belongs to
    stack_idx = tl.floor_div(c * C + block_c, C)
    actual_channel_idx = (c * C + block_c) % 256
    
    # Store all three stack elements
    out_offsets = (stack_idx * N + n) * (256 * 40 * 40) + actual_channel_idx * (40 * 40) + h * 40 + w
    
    # Store based on which stack index we're processing
    if stack_idx < 3:
        final_mask = masks & (actual_channel_idx < 256)
        if stack_idx == 0:  # tmp_1: interpolated in_0
            tl.store(out_ptr + out_offsets, in_0_val, mask=final_mask)
        elif stack_idx == 1:  # tmp_2: interpolated in_1  
            tl.store(out_ptr + out_offsets, in_1_val, mask=final_mask)
        else:  # tmp_0: concatenated in_2, in_3
            final_mask2 = final_mask & ((c_in2 < C2) | (c_in3 >= 0 & c_in3 < C2))
            tl.store(out_ptr + out_offsets, concat_val, mask=final_mask2)

@torch.fx.wrap  
def fused_stack_operation(in_0, in_1, in_2, in_3):
    N, C, H, W = in_0.shape
    C2, _, _, _ = in_2.shape
    
    # Output shape: [3, N, C+C2, 40, 40]
    out_shape = (3, N, C + C2, 40, 40)
    out = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Block size for channels
    BLOCK_M = 32  # Process 32 channels at a time for better GPU utilization
    
    # Grid dimensions: [N, 3, 512//BLOCK_M, 40, 40] - process all 3 stacks at once
    grid = (
        N,
        3,  # Three stack elements
        (512 + BLOCK_M - 1) // BLOCK_M,  # Number of channel blocks
        40,
        40
    )
    
    fused_stack_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=out,
        N=N,
        C=C,
        C2=C2,
        BLOCK_M=BLOCK_M
    )
    
    return out

def pattern(x, y):
    return torch.cat((x, y), 1)

def replacement_args(*args):
    # Return all arguments as-is
    return args

@triton.jit
def simple_concat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    x_size: tl.constexpr,
    y_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < x_size + y_size
    
    # Copy x to first part of output
    x_mask = offsets < x_size
    x_vals = tl.load(x_ptr + offsets, mask=x_mask, other=0.0)
    tl.store(out_ptr + offsets, x_vals, mask=x_mask)
    
    # Copy y to second part of output
    y_mask = (offsets >= x_size) & mask
    y_offsets = offsets - x_size
    y_vals = tl.load(y_ptr + y_offsets, mask=y_mask, other=0.0)
    tl.store(out_ptr + offsets, y_vals, mask=y_mask)

@torch.fx.wrap
def simple_concat(x, y):
    """Optimized concatenation kernel using Triton"""
    x_size = x.numel()
    y_size = y.numel()
    total_size = x_size + y_size
    
    out = torch.empty(total_size, dtype=x.dtype, device=x.device)
    
    # Use appropriate block size for better GPU utilization
    BLOCK_SIZE = 1024
    num_programs = (total_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Grid dimensions: [num_programs] 
    grid = (num_programs,)
    
    simple_concat_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        x_size=x_size,
        y_size=y_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out.view(x.shape[0], x.shape[1] + y.shape[1], x.shape[2], x.shape[3])

def replacement_func():
    return simple_concat