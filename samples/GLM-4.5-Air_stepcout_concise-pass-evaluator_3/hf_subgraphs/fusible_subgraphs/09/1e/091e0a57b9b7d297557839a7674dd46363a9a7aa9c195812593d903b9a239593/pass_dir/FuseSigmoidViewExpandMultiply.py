import torch
import triton
import triton.language as tl
import math

# Pattern matching function
def pattern(in_0, in_1, in_2):
    """Match sigmoid + view + expand + multiply pattern"""
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    tmp_3 += in_0
    tmp_4 = tmp_3
    tmp_5 = torch.nn.functional.relu(tmp_4, inplace=True)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7  # Return the final observable output

# Argument extraction function  
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized Triton kernel
@triton.jit
def fused_full_computation_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, out_ptr,
    N: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Fused kernel that performs the entire computation: sigmoid -> multiply -> add -> relu -> pool -> flatten"""
    pid = tl.program_id(0)
    
    # For the flattened output, we only need N x C elements (after pooling to 1x1)
    total_output_elements = N * C
    output_block_start = pid * BLOCK_SIZE
    output_offsets = output_block_start + tl.arange(0, BLOCK_SIZE)
    output_mask = output_offsets < total_output_elements
    
    # Calculate which output element this program handles
    output_idx = tl.load(output_offsets, mask=output_mask)
    if not output_mask:
        return
        
    # Map output index back to input tensor indices
    # Output shape: [N, C] -> input shape [N, C, H, W]
    n_idx = output_idx // C
    c_idx = output_idx % C
    
    # Load the three inputs at position [n_idx, c_idx, :, :] (the entire channel for this batch)
    # We need to handle the different shapes:
    # in_0, in_1: [N, C, H, W] 
    # in_2: [N, 1, C]
    
    # For in_2, we only need the value at [n_idx, 0, c_idx] and broadcast it
    if n_idx * C + c_idx < in_2_ptr.numel():
        in_2_val = tl.load(in_2_ptr + n_idx * C + c_idx)
    else:
        in_2_val = 0.0  # fallback
    
    # Apply sigmoid to in_2 value (replaces: tmp_0 = in_2.sigmoid())
    sigmoid_val = 1.0 / (1.0 + tl.exp(-in_2_val))
    
    # Load in_0 and in_1 for this channel and broadcast sigmoid_val across H x W
    # We'll process one channel at a time
    sum_val = 0.0
    
    # Create 2D indices for H x W
    h_indices = tl.arange(0, H)
    w_indices = tl.arange(0, W)
    h_grid, w_grid = tl.meshgrid(h_indices, w_indices)
    
    # Flatten the 2D indices for 1D access
    flat_hw_indices = h_grid * W + w_grid
    
    # Process all H x W positions for this channel
    for hw_idx in range(H * W):
        if hw_idx >= BLOCK_SIZE:
            break
            
        offset_idx = output_block_start + hw_idx
        if offset_idx >= N * C * H * W:
            break
            
        # Load in_0 and in_1 values at this position
        in_0_val = tl.load(in_0_ptr + offset_idx, other=0.0)
        in_1_val = tl.load(in_1_ptr + offset_idx, other=0.0)
        
        # Apply the computation: sigmoid_val * in_1 + in_0, then ReLU
        mul_result = sigmoid_val * in_1_val
        add_result = mul_result + in_0_val
        relu_result = tl.maximum(add_result, 0.0)
        
        # Accumulate for average pooling (sum all values)
        sum_val += relu_result
    
    # Apply average pooling (divide by H * W) and store result
    pooled_result = sum_val / (H * W)
    tl.store(out_ptr + output_idx, pooled_result, mask=output_mask)

# Kernel wrapper
@torch.fx.wrap
def fused_sigmoid_view_expand_multiply(in_0, in_1, in_2):
    N, C, H, W = in_1.shape
    
    # Set up block size and grid for final output [N, C]
    BLOCK_SIZE = 1024
    total_output_elements = N * C
    num_programs = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor (flattened from adaptive_avg_pool2d result)
    out = torch.empty(N * C, dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel
    fused_full_computation_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return fused_sigmoid_view_expand_multiply