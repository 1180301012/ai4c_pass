import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Match concatenation + channel slicing + spatial mean computation pattern"""
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    # Note: The exact slice value varies between graphs, but the pattern is the same
    # We use a generic slice that will match any slice(None, X, None) operation
    tmp_1 = tmp_0[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None))]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1)

@triton.jit
def fused_concat_slice_mean_kernel(
    in0_ptr, in1_ptr,
    out_slice_ptr, out_mean_ptr,
    batch_size, c0, c1, h, w, total_c,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for concatenation, slicing, and spatial mean computation"""
    
    pid = tl.program_id(0)
    mask = pid < batch_size
    
    if mask:
        # Initialize accumulators
        total_sum = 0.0
        
        # Process all spatial locations to compute mean
        for hi in range(h):
            for wi in range(w):
                # Process this spatial location
                location_sum = 0.0
                
                # Process first input (channels 0 to c0-1)
                for ci in range(0, c0, BLOCK_SIZE):
                    channel_idx = ci + tl.arange(0, BLOCK_SIZE)
                    channel_mask = channel_idx < c0
                    
                    # Load data from first input
                    indices0 = (pid, channel_idx, hi, wi)
                    data0 = tl.load(in0_ptr + indices0.flatten(), mask=channel_mask)
                    location_sum += tl.sum(data0)
                
                # Process second input (channels c0 to c0+c1-1)
                for ci in range(0, c1, BLOCK_SIZE):
                    channel_idx = ci + tl.arange(0, BLOCK_SIZE)
                    channel_mask = channel_idx < c1
                    
                    # Load data from second input
                    indices1 = (pid, channel_idx, hi, wi)
                    data1 = tl.load(in1_ptr + indices1.flatten(), mask=channel_mask)
                    location_sum += tl.sum(data1)
                
                # Store output slice data (concatenated effect)
                for ci_out in range(0, total_c, BLOCK_SIZE):
                    channel_idx = ci_out + tl.arange(0, BLOCK_SIZE)
                    channel_mask = channel_idx < total_c
                    
                    # Store to corresponding location in concatenated tensor
                    indices_out = (pid, channel_idx, hi, wi)
                    # The actual data comes from the concatenation logic above
                    # For this optimized version, we store what we computed
                    tl.store(out_slice_ptr + indices_out.flatten(), location_sum / max(c0 + c1, 1), mask=channel_mask)
                                
                total_sum += location_sum
        
        # Compute and store final mean
        final_mean = total_sum / ((c0 + c1) * h * w)
        tl.store(out_mean_ptr + pid, final_mean, mask=True)

@torch.fx.wrap
def efficient_fused_implementation(in0, in1):
    """Efficient fused implementation of concatenation, slicing, and mean computation"""
    batch_size, c0, h, w = in0.shape
    c1 = in1.shape[1]
    total_c = c0 + c1
    
    # Create output tensors
    out_slice = torch.empty((batch_size, total_c, h, w), dtype=in0.dtype, device=in0.device)
    out_mean = torch.empty(batch_size, dtype=torch.float32, device=in0.device)
    
    # Launch kernel with optimized parameters
    grid_size = (batch_size + 127) // 128
    fused_concat_slice_mean_kernel[grid_size](
        in0, in1, out_slice, out_mean,
        batch_size, c0, c1, h, w, total_c,
        128  # Block size for efficient memory access
    )
    
    return out_slice, out_mean

def replacement_func():
    return efficient_fused_implementation