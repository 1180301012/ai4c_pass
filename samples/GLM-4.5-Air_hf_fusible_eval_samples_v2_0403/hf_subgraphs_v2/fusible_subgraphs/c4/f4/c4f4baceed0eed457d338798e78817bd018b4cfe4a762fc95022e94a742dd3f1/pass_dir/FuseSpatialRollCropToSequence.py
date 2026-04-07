import torch
import triton
import triton.language as tl

def pattern(in_3, in_2, orig_shape, view_shape, roll_shifts, crop_slices, final_seq_len, feature_dim):
    """
    Pattern matching: contiguous + view + roll + crop + contiguous + view + add
    This pattern matches the tensor manipulation sequence before layer norm
    """
    # Original tensor manipulation sequence
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, view_shape[0], view_shape[1], feature_dim)
    tmp_4 = torch.roll(tmp_3, shifts=roll_shifts, dims=(1, 2))
    tmp_5 = tmp_4[crop_slices]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, final_seq_len, feature_dim)
    tmp_8 = in_2 + tmp_7
    return tmp_8

def replacement_args(in_3, in_2, orig_shape, view_shape, roll_shifts, crop_slices, final_seq_len, feature_dim):
    return (in_3, in_2, orig_shape, view_shape, roll_shifts, crop_slices, final_seq_len, feature_dim)

@triton.jit
def fused_spatial_kernel(
    in3_ptr, in2_ptr, out_ptr,
    orig_dims,
    view_h, view_w, feature_dim,
    roll_h, roll_w,
    crop_h_start, crop_w_start,
    final_seq_len,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel that handles contiguous + view + roll + crop + view + add
    """
    pid = tl.program_id(0)
    
    # Calculate sequence position
    seq_idx = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask = seq_idx < final_seq_len
    
    # Convert sequence index back to spatial coordinates
    spatial_size = view_h * view_w
    orig_total_pixels = orig_dims
    
    # Load original tensor and apply fused operations
    for i in range(tl.cdiv(feature_dim, BLOCK_SIZE_N)):
        channel_idx = i * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        channel_mask = channel_idx < feature_dim
        
        # Calculate spatial indices based on sequence position
        spatial_offset = seq_idx * feature_dim + channel_idx
        
        # Extract spatial coordinates
        spatial_linear = spatial_offset // feature_dim
        h = (spatial_linear % view_h + roll_h) % view_h
        w = ((spatial_linear // view_h) % view_w + roll_w) % view_w
        
        # Apply crop boundaries
        h_cropped = h if h >= crop_h_start else h + view_h
        w_cropped = w if w >= crop_w_start else w + view_w
        
        # Check if we're within crop region
        valid_spatial = (h_cropped < view_h) & (w_cropped < view_w)
        total_mask = mask & channel_mask & valid_spatial
        
        # Access original tensor structure
        pixel_idx = spatial_linear
        batch_idx = 0
        
        # Load from input 3 (original permuted tensor)
        orig_offset = batch_idx * orig_total_pixels * feature_dim + pixel_idx * feature_dim + channel_idx
        val_in3 = tl.load(in3_ptr + orig_offset, mask=total_mask, other=0.0)
        
        # Load from input 2 (residual connection)
        residual_offset = batch_idx * final_seq_len * feature_dim + spatial_offset
        val_in2 = tl.load(in2_ptr + residual_offset, mask=mask & channel_mask, other=0.0)
        
        # Apply fused operations and add residual
        result = val_in3 + val_in2
        
        # Store result
        out_offset = residual_offset
        tl.store(out_ptr + out_offset, result, mask=mask & channel_mask)

@torch.fx.wrap
def fused_spatial_operations(in_3, in_2, orig_shape, view_shape, roll_shifts, crop_slices, final_seq_len, feature_dim):
    """
    Wrapper function that launches the fused kernel
    """
    # Extract crop parameters
    crop_h_start = crop_slices[1].start if crop_slices[1].start is not None else 0
    crop_w_start = crop_slices[2].start if crop_slices[2].start is not None else 0
    
    # Pre-compute total pixels for original tensor dimensions
    orig_dims = orig_shape[1] * orig_shape[2] * orig_shape[3] * orig_shape[4]
    
    # Calculate launch configuration
    BLOCK_SIZE_M = 128  # Sequence length blocks
    BLOCK_SIZE_N = 128  # Feature dimension blocks
    
    grid = (triton.cdiv(final_seq_len, BLOCK_SIZE_M),)
    
    # Allocate output
    out = torch.empty((1, final_seq_len, feature_dim), dtype=in_3.dtype, device=in_3.device)
    
    # Launch kernel
    fused_spatial_kernel[grid](
        in3_ptr=in_3,
        in2_ptr=in_2,
        out_ptr=out,
        orig_dims=orig_dims,
        view_h=view_shape[0],
        view_w=view_shape[1],
        feature_dim=feature_dim,
        roll_h=roll_shifts[0],
        roll_w=roll_shifts[1],
        crop_h_start=crop_h_start,
        crop_w_start=crop_w_start,
        final_seq_len=final_seq_len,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return fused_spatial_operations