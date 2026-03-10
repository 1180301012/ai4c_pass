import torch
import triton
import triton.language as tl

def pattern(x, other):
    # Match: max_pool2d + interpolate + cat pattern
    # Use the exact parameter order from the graphs
    tmp_4 = torch.nn.functional.max_pool2d(x, 2, 2, 0, 1, False, False)
    # Use exact parameter order and style from graphs
    target_size = (other.shape[2], other.shape[3])
    tmp_5 = torch.nn.functional.interpolate(tmp_4, target_size, None, 'bilinear', False)
    tmp_6 = torch.cat([other, tmp_5], 1)
    return tmp_6

def replacement_args(x, other):
    return x, other

@triton.jit
def fused_pool_interp_cat_kernel(
    x_ptr,
    other_ptr,
    out_ptr,
    batch_size,
    channels_in,
    height_in,
    width_in,
    channels_other,
    height_target,
    width_target,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate batch and channel offsets
    batch_offset = pid_m * BLOCK_SIZE_M
    channel_offset = pid_n * BLOCK_SIZE_N
    
    # Ensure we don't go out of bounds
    if batch_offset >= batch_size:
        return
    
    # Load input tile
    x_height = height_in // 2
    x_width = width_in // 2
    
    # Process each position in the output
    for h in range(0, height_target, BLOCK_SIZE_M):
        for w in range(0, width_target, BLOCK_SIZE_N):
            h_out = h + batch_offset
            w_out = w + channel_offset
            
            if h_out >= height_target or w_out >= width_target:
                continue
                
            # For bilinear interpolation, sample 4 points from pooled input
            x_h = h_out * 2  # Scale factor 2x up
            x_w = w_out * 2
            
            # Clamp coordinates
            x_h = min(x_h, x_height - 1)
            x_w = min(x_w, x_width - 1)
            
            # Get bilinear weights
            h_frac = x_h - (x_h // 1)
            w_frac = x_w - (x_w // 1)
            
            h1 = 1.0 - h_frac
            w1 = 1.0 - w_frac
            h2 = h_frac
            w2 = w_frac
            
            # Sample from pooled input using bilinear interpolation
            # Top-left
            val_tl = tl.load(x_ptr + batch_offset * channels_in * x_height * x_width + 
                            0 * x_height * x_width + x_h * x_width + x_w)
            
            # Top-right (if not at boundary)
            if x_w + 1 < x_width:
                val_tr = tl.load(x_ptr + batch_offset * channels_in * x_height * x_width + 
                                0 * x_height * x_width + x_h * x_width + (x_w + 1))
            else:
                val_tr = val_tl
            
            # Bottom-left (if not at boundary)
            if x_h + 1 < x_height:
                val_bl = tl.load(x_ptr + batch_offset * channels_in * x_height * x_width + 
                                0 * x_height * x_width + (x_h + 1) * x_width + x_w)
            else:
                val_bl = val_tl
            
            # Bottom-right (if not at boundary)
            if x_h + 1 < x_height and x_w + 1 < x_width:
                val_br = tl.load(x_ptr + batch_offset * channels_in * x_height * x_width + 
                                0 * x_height * x_width + (x_h + 1) * x_width + (x_w + 1))
            else:
                val_br = val_tl
            
            # Bilinear interpolation
            interp_val = (h1 * w1 * val_tl + h1 * w2 * val_tr + 
                         h2 * w1 * val_bl + h2 * w2 * val_br)
            
            # Load from other tensor (first half)
            other_val = tl.load(other_ptr + batch_offset * channels_other * height_target * width_target + 
                              0 * height_target * width_target + h_out * width_target + w_out)
            
            # Concatenate: other (first), interpolated (second)
            out_val = other_val + interp_val  # Simple fusion, can be tailored
            
            tl.store(out_ptr + batch_offset * (channels_other + 1) * height_target * width_target + 
                    0 * height_target * width_target + h_out * width_target + w_out, 
                    out_val)

@torch.fx.wrap
def fused_pool_interp_cat(x, other):
    batch_size = x.shape[0]
    channels_in = x.shape[1]
    height_in = x.shape[2]
    width_in = x.shape[3]
    channels_other = other.shape[1]
    height_target = other.shape[2]
    width_target = other.shape[3]
    
    out = torch.empty(batch_size, channels_other + 1, height_target, width_target, 
                     device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    grid = (triton.cdiv(height_target, BLOCK_SIZE_M), 
            triton.cdiv(width_target, BLOCK_SIZE_N))
    
    fused_pool_interp_cat_kernel[grid](
        x, other, out,
        batch_size, channels_in, height_in, width_in,
        channels_other, height_target, width_target,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    return out

def replacement_func():
    return fused_pool_interp_cat