import torch
import triton
import triton.language as tl

@triton.jit
def bicubic_interpolate_mid_kernel(
    input_ptr, output_ptr, 
    n_channels, in_height, in_width, out_height, out_width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n_channels * out_height * out_width)
    
    # Convert offset to coordinates
    c = offsets // (out_height * out_width)
    out_y = (offsets // out_width) % out_height
    out_x = offsets % out_width
    
    # Calculate scale factors
    scale_y = in_height / out_height
    scale_x = in_width / out_width
    
    # Bicubic interpolation using 4x4 neighborhood
    for oc in range(n_channels):
        # Calculate input coordinates
        in_y = out_y * scale_y + 0.5
        in_x = out_x * scale_x + 0.5
        
        # Get integer coordinates and weights
        y0 = int(in_y - 1.5)
        y1 = int(in_y - 0.5)
        y2 = int(in_y + 0.5)
        y3 = int(in_y + 1.5)
        x0 = int(in_x - 1.5)
        x1 = int(in_x - 0.5)
        x2 = int(in_x + 0.5)
        x3 = int(in_x + 1.5)
        
        # Weights using cubic B-spline
        def weight(t):
            if abs(t) < 1:
                return 0.5 * abs(t)**3 - abs(t)**2 + 2/3
            else:
                return -0.5 * abs(t)**3 + 2.5 * abs(t)**2 - 4 * abs(t) + 8/3
        
        wy0, wy1, wy2, wy3 = weight(in_y - y0), weight(in_y - y1), weight(in_y - y2), weight(in_y - y3)
        wx0, wx1, wx2, wx3 = weight(in_x - x0), weight(in_x - x1), weight(in_x - x2), weight(in_x - x3)
        
        # Sum weighted neighbors
        sum_val = 0.0
        for yi in range(y0, y3 + 1):
            for xi in range(x0, x3 + 1):
                if 0 <= yi < in_height and 0 <= xi < in_width:
                    input_idx = oc * in_height * in_width + yi * in_width + xi
                    pixel = tl.load(input_ptr + input_idx, mask=True, other=0.0)
                    sum_val += pixel * wy[yi - y0] * wx[xi - x0]
        
        # Store output
        output_idx = offsets
        tl.store(output_ptr + output_idx, sum_val, mask=mask)

@torch.fx.wrap
def process_mid_position_embeddings(mid_pos_emb):
    # Extract the tensors as in the pattern
    slice_none = slice(None, None, None)
    slice_neg10 = slice(-10, None, None)
    slice_1_neg10 = slice(1, -10, None)
    
    tmp_25 = mid_pos_emb[(slice_none, slice_none, 0, slice_none)]
    tmp_26 = tmp_25[(slice_none, None)]
    tmp_27 = mid_pos_emb[(slice_none, slice_none, slice_neg10, slice_none)]
    tmp_28 = mid_pos_emb[(slice_none, slice_none, slice_1_neg10, slice_none)]
    tmp_29 = tmp_28.transpose(2, 3)
    tmp_30 = tmp_29.view(4, 32, 15, 15)
    
    # Apply bicubic interpolation to middle tokens
    interpolated = torch.empty(4, 32, 15, 15, dtype=tmp_30.dtype, device=tmp_30.device)
    bicubic_interpolate_mid_kernel[(interpolated.numel() + 1023) // 1024,](
        tmp_30, interpolated,
        32, 15, 15, 15, 15,  # channels, in_h, in_w, out_h, out_w
        1024
    )
    
    # Process interpolated middle tokens to final shape
    middle_processed = interpolated.flatten(2).transpose(1, 2)  # [4, 7200, 32]
    middle_processed = middle_processed.contiguous()  # Ensure memory contiguity
    middle_final = middle_processed.view(4, 1, 225, 32)  # Final view for return
    
    # Ensure tmp_26 has correct shape for return
    first_final = tmp_26.unsqueeze(1) if tmp_26.dim() == 2 else tmp_26
    
    return first_final, tmp_27, middle_final

def pattern(mid_pos_emb):
    slice_none = slice(None, None, None)
    slice_neg10 = slice(-10, None, None)
    slice_1_neg10 = slice(1, -10, None)
    
    tmp_25 = mid_pos_emb[(slice_none, slice_none, 0, slice_none)]
    tmp_26 = tmp_25[(slice_none, None)]
    tmp_27 = mid_pos_emb[(slice_none, slice_none, slice_neg10, slice_none)]
    tmp_28 = mid_pos_emb[(slice_none, slice_none, slice_1_neg10, slice_none)]
    tmp_29 = tmp_28.transpose(2, 3)
    tmp_30 = tmp_29.view(4, 32, 15, 15)
    
    return tmp_26, tmp_27, tmp_30

def replacement_args(mid_pos_emb):
    return (mid_pos_emb,)

def replacement_func():
    return process_mid_position_embeddings