import torch
import triton
import triton.language as tl

@triton.jit
def bicubic_interpolate_kernel(
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
        in_y = out_y * scale_y + 0.5  # Add 0.5 for proper centering
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

@triton.jit
def concatenate_kernel(
    ptr1, dim1, ptr2, dim2, ptr3, dim3, 
    output_ptr, total_dim,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_dim
    
    seq_idx = offsets // total_dim
    col_idx = offsets % total_dim
    
    # Load values from each tensor based on column index
    val = 0.0
    if col_idx < dim1:
        # First tensor
        idx = seq_idx * dim1 + col_idx
        val = tl.load(ptr1 + idx, mask=mask, other=0.0)
    elif col_idx < dim1 + dim2:
        # Second tensor
        col_idx_second = col_idx - dim1
        idx = seq_idx * dim2 + col_idx_second
        val = tl.load(ptr2 + idx, mask=mask, other=0.0)
    else:
        # Third tensor
        col_idx_third = col_idx - dim1 - dim2
        idx = seq_idx * dim3 + col_idx_third
        val = tl.load(ptr3 + idx, mask=mask, other=0.0)
    
    tl.store(output_ptr + offsets, val, mask=mask)

@torch.fx.wrap
def process_position_embeddings(pos_emb):
    # Extract the tensors as in the pattern
    slice_none = slice(None, None, None)
    slice_neg10 = slice(-10, None, None)
    slice_1_neg10 = slice(1, -10, None)
    
    tmp_13 = pos_emb[(slice_none, 0, slice_none)]
    tmp_14 = tmp_13[(slice_none, None)]
    tmp_15 = pos_emb[(slice_none, slice_neg10, slice_none)]
    tmp_16 = pos_emb[(slice_none, slice_1_neg10, slice_none)]
    tmp_17 = tmp_16.transpose(1, 2)
    tmp_18 = tmp_17.view(1, 32, 15, 15)
    
    # Apply bicubic interpolation to middle tokens
    interpolated = torch.empty(1, 32, 15, 15, dtype=tmp_18.dtype, device=tmp_18.device)
    bicubic_interpolate_kernel[(interpolated.numel() + 1023) // 1024,](
        tmp_18, interpolated,
        32, 15, 15, 15, 15,  # channels, in_h, in_w, out_h, out_w
        1024
    )
    
    # Process interpolated middle tokens
    middle_processed = interpolated.flatten(2).transpose(1, 2)  # [1, 7200, 32]
    
    # Flatten first token for concatenation
    first_flat = tmp_14.flatten(1).transpose(0, 1)  # [1, 32] -> [1, 32]
    first_expanded = first_flat.expand(7200, -1)  # [7200, 32]
    
    # Flatten last tokens for concatenation  
    last_flat = tmp_15.flatten(1).transpose(0, 1)  # [1, 320] -> [1, 320]
    last_expanded = last_flat.expand(7200, -1)  # [7200, 320]
    
    # Concatenate all three parts: [7200, 32+1+320] = [7200, 353]
    final_result = torch.empty(7200, 353, dtype=first_expanded.dtype, device=first_expanded.device)
    
    # Use Triton kernel for concatenation
    concatenate_kernel[(final_result.numel() + 1023) // 1024,](
        first_expanded, 32,
        middle_processed, 1,
        last_expanded, 320,
        final_result,
        7200,
        1024
    )
    
    return final_result

def pattern(pos_emb):
    # Extract position embeddings
    tmp_13 = pos_emb[(slice(None, None, None), 0, slice(None, None, None))]
    tmp_14 = tmp_13[(slice(None, None, None), None)]
    tmp_15 = pos_emb[(slice(None, None, None), slice(-10, None, None), slice(None, None, None))]
    tmp_16 = pos_emb[(slice(None, None, None), slice(1, -10, None), slice(None, None, None))]
    tmp_17 = tmp_16.transpose(1, 2)
    tmp_18 = tmp_17.view(1, 32, 15, 15)
    
    # Return the extracted tensors before interpolation
    return tmp_14, tmp_15, tmp_18

def replacement_args(pos_emb):
    return (pos_emb,)

def replacement_func():
    return process_position_embeddings