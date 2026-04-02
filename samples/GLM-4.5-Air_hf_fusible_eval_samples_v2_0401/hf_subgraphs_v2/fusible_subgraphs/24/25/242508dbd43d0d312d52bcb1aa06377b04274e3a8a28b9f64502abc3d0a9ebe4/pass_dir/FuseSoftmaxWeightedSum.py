import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    """Match: softmax + element-wise multiply + sum along dim=1"""
    tmp_0 = torch.softmax(in_1, dim=1)
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_fused_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_channels,  # 256 
    height,      # varies: 14, 16, 21, 28
    width,       # varies: 14, 16, 21, 28,
    CHANNEL_BLOCK_SIZE: tl.constexpr,
    SPATIAL_BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized fused kernel with:
    - Channel blocking for better occupancy 
    - Spatial blocking for memory locality
    - Pre-computed softmax weights per channel
    """
    
    pid = tl.program_id(0)
    
    # Process channel blocks
    channel_start = pid * CHANNEL_BLOCK_SIZE
    channel_end = min((pid + 1) * CHANNEL_BLOCK_SIZE, n_channels)
    
    # Process each channel in this block
    for channel_idx in range(channel_start, channel_end):
        # Pre-compute softmax weights for this channel
        base_in1_idx = channel_idx * 2
        logit0 = tl.load(in_1_ptr + base_in1_idx + 0, mask=True, other=float('-inf'))
        logit1 = tl.load(in_1_ptr + base_in1_idx + 1, mask=True, other=float('-inf'))
        
        # Stable softmax computation
        max_val = tl.maximum(logit0, logit1)
        exp0 = tl.exp(logit0 - max_val)
        exp1 = tl.exp(logit1 - max_val)
        sum_exp = exp0 + exp1
        weight0 = exp0 / sum_exp
        weight1 = exp1 / sum_exp
        
        # Process spatial region for this channel
        h_start = 0
        h_end = height
        w_start = 0
        w_end = width
        
        # Process spatial blocks for memory coalescing
        for h in range(h_start, h_end, SPATIAL_BLOCK_SIZE):
            for w in range(w_start, w_end, SPATIAL_BLOCK_SIZE):
                # Process elements in this spatial block
                for dh in range(min(SPATIAL_BLOCK_SIZE, h_end - h)):
                    for dw in range(min(SPATIAL_BLOCK_SIZE, w_end - w)):
                        h_pos = h + dh
                        w_pos = w + dw
                        
                        # Linear output index
                        out_idx = channel_idx * height * width + h_pos * width + w_pos
                        
                        # Load input elements - efficient memory access pattern
                        elem_base = channel_idx * height * width + h_pos * width
                        elem0_ptr = in_0_ptr + 0 * (n_channels * height * width) + elem_base + w_pos
                        elem1_ptr = in_0_ptr + 1 * (n_channels * height * width) + elem_base + w_pos
                        
                        val0 = tl.load(elem0_ptr, mask=True, other=0.0)
                        val1 = tl.load(elem1_ptr, mask=True, other=0.0)
                        
                        # Compute weighted sum using pre-computed softmax weights
                        result = val0 * weight0 + val1 * weight1
                        tl.store(out_ptr + out_idx, result)

@torch.fx.wrap
def fused_softmax_weighted_sum(in_0, in_1):
    # Get input shapes
    batch_size, n_dim, channels, height, width = in_0.shape
    in_1_batch, in_1_dim, in_1_channels, _, _ = in_1.shape
    
    # Validate shapes
    assert batch_size == 1, f"Expected batch_size=1, got {batch_size}"
    assert in_1_batch == 1, f"Expected in_1_batch=1, got {in_1_batch}"
    assert n_dim == 2, f"Expected n_dim=2, got {n_dim}"
    assert in_1_dim == 2, f"Expected in_1_dim=2, got {in_1_dim}"
    assert channels == in_1_channels, f"Channel mismatch: {channels} vs {in_1_channels}"
    
    # Output shape after sum along dim=1: [1, 256, height, width]
    output_shape = [batch_size, channels, height, width]
    out = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Handle float16 by casting to float32 for computation, then back to float16
    orig_dtype = in_0.dtype
    if orig_dtype == torch.float16:
        in_0 = in_0.float()
        in_1 = in_1.float()
    
    # Optimized kernel parameters for better performance
    CHANNEL_BLOCK_SIZE = 16   # Process multiple channels per program for better occupancy
    SPATIAL_BLOCK_SIZE = 8    # Spatial block size for memory coalescing
    
    # Calculate total programs needed (one per channel block)
    num_programs = (channels + CHANNEL_BLOCK_SIZE - 1) // CHANNEL_BLOCK_SIZE
    
    # Launch optimized kernel
    optimized_fused_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1, 
        out_ptr=out,
        n_channels=channels,
        height=height,
        width=width,
        CHANNEL_BLOCK_SIZE=CHANNEL_BLOCK_SIZE,
        SPATIAL_BLOCK_SIZE=SPATIAL_BLOCK_SIZE,
    )
    
    # Convert back to original dtype if needed
    if orig_dtype == torch.float16:
        out = out.half()
    
    return out

def replacement_func():
    return fused_softmax_weighted_sum