import torch
import triton
import triton.language as tl

# Pattern matching function for Interpolate + Add fusion
def pattern(tmp_6, in_7):
    tmp_7 = torch.nn.functional.interpolate(tmp_6, (64, 64), None, 'bilinear', False)
    tmp_8 = in_7 + tmp_7
    return tmp_8  # Only return the observable value that's used downstream

# Argument extraction function
def replacement_args(tmp_6, in_7):
    return (tmp_6, in_7)

# Optimized kernel for fused Interpolate + Add
@triton.jit
def fused_interpolate_add_kernel(
    x_ptr,          # tmp_6: input tensor [B, C, H_in, W_in]
    bias_ptr,       # in_7: bias tensor [B, C, H_out, W_out]
    out_ptr,        # output temporary [B, C, H_out, W_out]
    B, C, H_in, W_in, H_out, W_out,  # Tensor dimensions
    BLOCK_SIZE_M: tl.constexpr,  # Block size for channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for spatial dimensions
):
    # Get program IDs
    pid_m = tl.program_id(0)  # Channel blocks
    pid_n = tl.program_id(1)  # Spatial blocks
    
    # Calculate ranges
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Create offsets for channels
    channel_offsets = m_offset + tl.arange(0, BLOCK_SIZE_M)
    channel_mask = channel_offsets < C
    
    # Process a block of spatial elements
    n_end = min(n_offset + BLOCK_SIZE_N, H_out * W_out)
    for spatial_idx in range(n_offset, n_end):
        # Target coordinates
        h_out = spatial_idx // W_out
        w_out = spatial_idx % W_out
        
        # Map to source coordinates (8x upscaling)
        scale_x = (W_in - 1) / (W_out - 1) if W_out > 1 else 0
        scale_y = (H_in - 1) / (H_out - 1) if H_out > 1 else 0
        
        x_in = tl.cast(h_out * scale_y, tl.float32)
        y_in = tl.cast(w_out * scale_x, tl.float32)
        
        # Get integer coordinates and weights for bilinear interpolation
        x0 = tl.floor(x_in)
        y0 = tl.floor(y_in)
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Clamp to bounds
        x0 = tl.maximum(x0, 0.0)
        y0 = tl.maximum(y0, 0.0)
        x1 = tl.minimum(x1, tl.cast(H_in - 1, tl.float32))
        y1 = tl.minimum(y1, tl.cast(W_in - 1, tl.float32))
        
        # Compute weights
        wa = (x1 - x_in) * (y1 - y_in)
        wb = (x1 - x_in) * (y_in - y0)
        wc = (x_in - x0) * (y1 - y_in)
        wd = (x_in - x0) * (y_in - y0)
        
        # Convert to integer indices
        x0_int = tl.cast(x0, tl.int32)
        y0_int = tl.cast(y0, tl.int32)
        x1_int = tl.cast(x1, tl.int32)
        y1_int = tl.cast(y1, tl.int32)
        
        # Load 4 neighboring values for each channel
        for channel_idx in range(BLOCK_SIZE_M):
            if m_offset + channel_idx >= C:
                break
                
            # Base pointer for current channel
            channel_base = channel_idx * H_in * W_in
            
            # Load corner values
            val_00 = tl.load(x_ptr + channel_base + x0_int * W_in + y0_int, mask=False)
            val_01 = tl.load(x_ptr + channel_base + x0_int * W_in + y1_int, mask=False)
            val_10 = tl.load(x_ptr + channel_base + x1_int * W_in + y0_int, mask=False)
            val_11 = tl.load(x_ptr + channel_base + x1_int * W_in + y1_int, mask=False)
            
            # Bilinear interpolation
            interpolated = wa * val_00 + wb * val_01 + wc * val_10 + wd * val_11
            
            # Load bias from second input tensor
            bias_offset = h_out * W_out + w_out
            bias_ptr_offset = channel_idx * H_out * W_out + bias_offset
            bias = tl.load(bias_ptr + bias_ptr_offset, mask=False)
            
            # Add bias to interpolated value
            out = interpolated + bias
            
            # Store result
            out_offset = h_out * W_out + w_out
            out_ptr_offset = channel_idx * H_out * W_out + out_offset
            tl.store(out_ptr + out_ptr_offset, out, mask=False)

@torch.fx.wrap
def fused_interpolate_add(tmp_6, in_7):
    # Get tensor shapes
    B, C, H_in, W_in = tmp_6.shape
    H_out, W_out = 64, 64  # Target size
    
    # Output shape should be [B, C, H_out, W_out]
    out = torch.empty((B, C, H_out, W_out), dtype=torch.float32, device=tmp_6.device)
    
    # Launch kernel
    BLOCK_SIZE_M = 64   # Number of channels to process per program
    BLOCK_SIZE_N = 256  # Number of spatial elements to process per program
    
    # Calculate grid size
    num_programs_m = (C + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_programs_n = (H_out * W_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    fused_interpolate_add_kernel[(num_programs_m, num_programs_n)](
        x_ptr=tmp_6,
        bias_ptr=in_7,
        out_ptr=out,
        B=B, C=C, H_in=H_in, W_in=W_in, H_out=H_out, W_out=W_out,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_interpolate_add