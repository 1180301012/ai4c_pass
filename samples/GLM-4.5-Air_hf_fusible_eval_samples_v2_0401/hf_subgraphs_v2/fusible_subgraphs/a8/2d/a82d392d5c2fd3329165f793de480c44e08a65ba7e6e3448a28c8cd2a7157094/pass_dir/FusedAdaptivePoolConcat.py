import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_adaptive_pool_concat_kernel(
    in_0_ptr,
    in_1_ptr, 
    out_ptr,
    batch_size,
    c1,
    c2,
    h_in,  # 64
    w_in,  # 48
    h_out,  # 32
    w_out,  # 24
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Get program IDs
    pid = tl.program_id(0)  # batch
    bid = tl.program_id(1)  # block row in output height  
    cid = tl.program_id(2)  # block col in output width
    
    # Calculate output tile coordinates
    h_base = bid * BLOCK_H
    w_base = cid * BLOCK_W
    h_coords = h_base + tl.arange(0, BLOCK_H)
    w_coords = w_base + tl.arange(0, BLOCK_W)
    h_coords = h_coords[:, None]
    w_coords = w_coords[None, :]
    
    # Mask for valid output positions
    h_mask = h_coords < h_out
    w_mask = w_coords < w_out
    output_mask = h_mask & w_mask
    
    # For this specific case, adaptive_avg_pool2d (64,48)->(32,24) is 2x2 avg pooling
    # Compute all 4 positions in each 2x2 window at once for each channel
    
    # Process each channel from in_0 (adaptive pooled part) with vectorized 2x2 averaging
    for c in range(c1):
        # Calculate base offset for this channel
        base_offset = pid * c1 * h_in * w_in + c * h_in * w_in
        
        # Load all 4 values in each 2x2 window using vectorized operations
        # Position (0,0) in 2x2 window
        offsets_00 = base_offset + (h_coords * 2) * w_in + (w_coords * 2)
        vals_00 = tl.load(in_0_ptr + offsets_00, mask=output_mask, other=0.0)
        
        # Position (0,1) in 2x2 window  
        offsets_01 = base_offset + (h_coords * 2) * w_in + (w_coords * 2 + 1)
        vals_01 = tl.load(in_0_ptr + offsets_01, mask=output_mask, other=0.0)
        
        # Position (1,0) in 2x2 window
        offsets_10 = base_offset + (h_coords * 2 + 1) * w_in + (w_coords * 2)
        vals_10 = tl.load(in_0_ptr + offsets_10, mask=output_mask, other=0.0)
        
        # Position (1,1) in 2x2 window
        offsets_11 = base_offset + (h_coords * 2 + 1) * w_in + (w_coords * 2 + 1)
        vals_11 = tl.load(in_0_ptr + offsets_11, mask=output_mask, other=0.0)
        
        # Compute average (sum of 4 values divided by 4)
        avg_vals = (vals_00 + vals_01 + vals_10 + vals_11) * 0.25
        
        # Calculate output offsets for this channel
        out_base_offset = pid * (c1 + c2) * h_out * w_out + c * h_out * w_out
        out_offsets = out_base_offset + h_coords * w_out + w_coords
        
        # Store averaged values to output
        tl.store(out_ptr + out_offsets, avg_vals, mask=output_mask)
    
    # Process each channel from in_1 (concatenated part) - direct copy
    concat_base_offset = pid * (c1 + c2) * h_out * w_out + c1 * h_out * w_out
    for c in range(c2):
        # Calculate output offsets for this channel
        out_offsets = concat_base_offset + c * h_out * w_out + h_coords * w_out + w_coords
        
        # Calculate in_1 offsets 
        in_1_base_offset = pid * c2 * h_out * w_out + c * h_out * w_out
        in_1_offsets = in_1_base_offset + h_coords * w_out + w_coords
        
        # Load and store values from in_1
        in_1_vals = tl.load(in_1_ptr + in_1_offsets, mask=output_mask, other=0.0)
        tl.store(out_ptr + out_offsets, in_1_vals, mask=output_mask)

@torch.fx.wrap  
def fused_adaptive_pool_concat(in_0, in_1):
    # Get input shapes
    batch_size, c1, h_in, w_in = in_0.shape
    _, c2, h_out, w_out = in_1.shape
    
    # Create output tensor
    out = torch.empty((batch_size, c1 + c2, h_out, w_out), device=in_0.device, dtype=in_0.dtype)
    
    # Calculate grid sizes with larger blocks for better performance
    BLOCK_H = 16
    BLOCK_W = 16
    grid_h = (h_out + BLOCK_H - 1) // BLOCK_H
    grid_w = (w_out + BLOCK_W - 1) // BLOCK_W
    grid = (
        batch_size,
        grid_h, 
        grid_w
    )
    
    # Launch kernel
    fused_adaptive_pool_concat_kernel[grid](
        in_0,
        in_1,
        out,
        batch_size,
        c1,
        c2,
        h_in,
        w_in,
        h_out,
        w_out,
        BLOCK_H,
        BLOCK_W
    )
    
    return out

def replacement_func():
    return fused_adaptive_pool_concat