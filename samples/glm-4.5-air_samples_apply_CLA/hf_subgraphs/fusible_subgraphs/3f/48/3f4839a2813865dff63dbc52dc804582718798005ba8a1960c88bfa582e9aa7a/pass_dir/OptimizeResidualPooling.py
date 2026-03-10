import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # tmp_2 = torch.nn.functional.avg_pool2d(in_2, 3, 1, 1, False, False, None)
    tmp_2 = torch.nn.functional.avg_pool2d(in_2, 3, 1, 1, False, False, None)
    # tmp_3 = tmp_2 - in_2  
    tmp_3 = tmp_2 - in_2
    # tmp_4 = in_0.unsqueeze(-1)
    tmp_4 = in_0.unsqueeze(-1)
    # tmp_5 = tmp_4.unsqueeze(-1)
    tmp_5 = tmp_4.unsqueeze(-1)
    # tmp_6 = tmp_5 * tmp_3
    tmp_6 = tmp_5 * tmp_3
    # tmp_7 = in_2 + tmp_6
    tmp_7 = in_2 + tmp_6
    # tmp_8 = in_1.unsqueeze(-1)
    tmp_8 = in_1.unsqueeze(-1)
    # tmp_9 = tmp_8.unsqueeze(-1)
    tmp_9 = tmp_8.unsqueeze(-1)
    
    return tmp_7, tmp_9

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_residual_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, out_7_ptr, out_9_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel
    pid = tl.program_id(0)
    
    if pid >= C:
        return
    
    # Load layer scale values
    layer_scale_1 = tl.load(in_0_ptr + pid, other=0.0)
    layer_scale_2 = tl.load(in_1_ptr + pid, other=0.0)
    
    # Process spatial positions in blocks
    spatial_size = H * W
    
    # Use vectorized loads for better performance
    for spatial_idx in range(0, spatial_size, BLOCK_SIZE):
        # Compute mask for this block
        mask = spatial_idx + tl.arange(0, BLOCK_SIZE) < spatial_size
        
        # Load current input values for this channel across all spatial positions
        input_vals = tl.load(in_2_ptr + pid * spatial_size + spatial_idx, mask=mask, other=0.0)
        
        # Compute average pooling with optimized window access
        pooled_vals = tl.zeros(BLOCK_SIZE, dtype=tl.float32)
        pool_counts = tl.zeros(BLOCK_SIZE, dtype=tl.int32)
        
        # Optimized 3x3 pooling with better memory access
        for kh in range(-1, 2):
            for kw in range(-1, 2):
                # Calculate neighbor offset
                neighbor_offset = (kh * W + kw)
                neighbor_idx = spatial_idx + neighbor_offset
                
                # Only load if neighbors are in bounds
                neighbor_mask = (neighbor_idx >= 0) & (neighbor_idx < spatial_size)
                combined_mask = mask & neighbor_mask
                
                if tl.any(combined_mask):
                    neighbor_vals = tl.load(in_2_ptr + pid * spatial_size + neighbor_idx, 
                                          mask=combined_mask, other=0.0)
                    pooled_vals += neighbor_vals
                    pool_counts += tl.int32(combined_mask)
        
        # Avoid division by zero
        safe_counts = tl.maximum(pool_counts, tl.ones_like(pool_counts))
        pooled_vals = pooled_vals / safe_counts
        
        # Compute fused operations: in_2 + scale_1 * (pooled - in_2)
        # This can be simplified to: in_2 * (1 - scale_1) + scale_1 * pooled
        residual = pooled_vals - input_vals
        scaled_residual = layer_scale_1 * residual
        output_vals = input_vals + scaled_residual
        
        # Store results
        tl.store(out_7_ptr + pid * spatial_size + spatial_idx, output_vals, mask=mask)
    
    # Expand layer scale 2 for output (this is cheap, do in host code)
    # Note: This part is simplified for now - in production would optimize this too

@torch.fx.wrap
def fused_residual_wrapper(in_0, in_1, in_2):
    N, C, H, W = in_2.shape
    
    # Create output tensors
    output_7 = torch.empty_like(in_2)
    output_9 = torch.empty((C, H, W), dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel
    grid = (triton.cdiv(C, 1),)
    
    # Use optimal block size for this hardware
    fused_residual_kernel[grid](
        in_0, in_1, in_2, output_7, output_9,
        N, C, H, W,
        BLOCK_SIZE=256,  # Optimized for NVIDIA A30
    )
    
    # Handle the tensor expansion on host for now (this is minor overhead)
    output_9 = in_1.unsqueeze(-1).unsqueeze(-1)
    
    return output_7, output_9

def replacement_func():
    return fused_residual_wrapper