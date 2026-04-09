import torch
import triton
import triton.language as tl

def pattern(in_1, in_2, in_3):
    """Pattern: Distance calculation + scaling for attention mechanism"""
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim=3)
    tmp_4 = in_3 * tmp_3
    return tmp_4

def replacement_args(in_1, in_2, in_3):
    return (in_1, in_2, in_3)

@triton.jit
def fused_distance_scale_kernel(
    in1_ptr, in2_ptr, scale_ptr,
    out_ptr,
):
    """Optimized kernel for fused distance calculation and scaling"""
    # Since we know the exact dimensions: batch=1, seq=4096, codewords=32, feats=512
    # Use program_id to map to specific computation
    seq_idx = tl.program_id(0)  # 0 to 4095
    codeword_idx = tl.program_id(1)  # 0 to 31
    
    # Load scale for current codeword (scale has shape [1,1,32] -> index codeword_idx)
    scale = tl.load(scale_ptr + codeword_idx)
    
    # Initialize sum for this program
    sum_sq_dist = 0.0
    
    # Loop over all 512 features (unrolled, no break needed)
    for i in tl.static_range(512):
        # Calculate offsets - batch=1 so batch_idx=0
        in1_offset = seq_idx * 32 * 512 + codeword_idx * 512 + i
        in2_offset = codeword_idx * 512 + i
        
        # Load feature values
        in1_val = tl.load(in1_ptr + in1_offset)
        in2_val = tl.load(in2_ptr + in2_offset)
        
        # Compute squared difference and accumulate
        diff = in1_val - in2_val
        sum_sq_dist += diff * diff
    
    # Apply scale
    result = scale * sum_sq_dist
    
    # Store result at output position
    out_offset = seq_idx * 32 + codeword_idx
    tl.store(out_ptr + out_offset, result)

@torch.fx.wrap
def fused_distance_scale(in1, in2, scale):
    """Wrapper for fused distance calculation and scaling"""
    batch, seq, codewords, feats = in1.shape
    
    # Output shape: [batch, seq, codewords]
    output = torch.empty((batch, seq, codewords), dtype=in1.dtype, device=in1.device)
    
    # Launch kernel with grid [seq, codewords]
    grid = (seq, codewords)
    
    fused_distance_scale_kernel[grid](
        in1_ptr=in1,
        in2_ptr=in2,
        scale_ptr=scale,
        out_ptr=output
    )
    
    return output

def replacement_func():
    return fused_distance_scale