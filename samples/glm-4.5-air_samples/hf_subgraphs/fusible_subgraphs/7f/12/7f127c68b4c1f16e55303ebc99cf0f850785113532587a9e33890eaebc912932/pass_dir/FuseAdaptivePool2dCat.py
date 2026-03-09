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
def fused_adaptive_pool2d_cat_kernel(
    x_ptr,           # Input tensor pointer [B, C, H, W]
    y_ptr,           # Second tensor pointer [B, C2, H_out, W_out]
    out_ptr,         # Output tensor pointer [B, C+C2, H_out, W_out]
    batch_size,      # Batch size
    in_channels,     # Input channels
    out_channels,    # Second tensor channels
    in_height,       # Input height
    in_width,        # Input width
    out_height,      # Output height (32)
    out_width,       # Output width (24)
    BLOCK_SIZE_M: tl.constexpr,    # Block size for M (channels)
    BLOCK_SIZE_N: tl.constexpr,    # Block size for N (height*width)
):
    # Program identifiers
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)  # Batch dimension
    
    # Calculate ranges for this program
    m_range_start = pid_m * BLOCK_SIZE_M
    n_range_start = pid_n * BLOCK_SIZE_N
    b = pid_b
    
    # Process spatial positions within this block
    h_out_base = (n_range_start // out_width)
    w_out_base = (n_range_start % out_width)
    
    # Create vectorized offsets for spatial processing
    h_offsets = h_out_base + tl.arange(0, BLOCK_SIZE_N)
    w_offsets = w_out_base + tl.arange(0, BLOCK_SIZE_N)
    h_mask = h_offsets < out_height
    w_mask = w_offsets < out_width
    spatial_mask = h_mask[:, None] & w_mask[None, :]
    
    # Process pooling part (first in_channels)
    if m_range_start < in_channels:
        # Calculate pooling scale factors
        h_scale = in_height / out_height
        w_scale = in_width / out_width
        
        # Process each channel in this block
        for c_local in tl.static_range(BLOCK_SIZE_M):
            c_actual = m_range_start + c_local
            if c_actual < in_channels:
                # Calculate output base offset for this channel
                out_base = b * (in_channels + out_channels) * out_height * out_width + c_actual * out_height * out_width
                
                # Vectorized processing of spatial positions
                for h_idx in h_offsets:
                    if h_idx < out_height:
                        for w_idx in w_offsets:
                            if w_idx < out_width:
                                # Calculate corresponding input region bounds
                                h_start = tl.cast(h_idx * h_scale, tl.int32)
                                h_end = tl.cast((h_idx + 1) * h_scale, tl.int32)
                                if h_end > in_height:
                                    h_end = in_height
                                w_start = tl.cast(w_idx * w_scale, tl.int32)
                                w_end = tl.cast((w_idx + 1) * w_scale, tl.int32)
                                if w_end > in_width:
                                    w_end = in_width
                                
                                # Load and compute average for this output position
                                if h_end > h_start and w_end > w_start:
                                    # Calculate total elements for averaging
                                    total_elements = (h_end - h_start) * (w_end - w_start)
                                    
                                    # Sum over the input region (vectorized inner loop)
                                    sum_val = 0.0
                                    for h_off in range(h_start, h_end):
                                        for w_off in range(w_start, w_end):
                                            # Load input element
                                            x_val = tl.load(
                                                x_ptr + b * in_channels * in_height * in_width + 
                                                c_actual * in_height * in_width + 
                                                h_off * in_width + w_off
                                            )
                                            sum_val += x_val
                                    
                                    # Compute average and store
                                    pool_val = sum_val / total_elements
                                    tl.store(
                                        out_base + h_idx * out_width + w_idx,
                                        pool_val
                                    )
    
    # Process concatenation part (second tensor)
    if m_range_start >= in_channels:
        # Process each channel in this block for second tensor
        for c_local in tl.static_range(BLOCK_SIZE_M):
            c_y_actual = m_range_start - in_channels + c_local
            if c_y_actual < out_channels:
                # Calculate output base offset for this channel (concatenation position)
                out_base = b * (in_channels + out_channels) * out_height * out_width + (in_channels + c_y_actual) * out_height * out_width
                
                # Vectorized processing of spatial positions
                for h_idx in h_offsets:
                    if h_idx < out_height:
                        for w_idx in w_offsets:
                            if w_idx < out_width:
                                # Load from second tensor
                                y_val = tl.load(
                                    y_ptr + b * out_channels * out_height * out_width + 
                                    c_y_actual * out_height * out_width + 
                                    h_idx * out_width + w_idx
                                )
                                
                                # Store to output at concatenation position
                                tl.store(
                                    out_base + h_idx * out_width + w_idx,
                                    y_val
                                )
                        

@torch.fx.wrap
def fused_adaptive_pool2d_cat(in_0, in_1):
    # Extract tensor shapes and properties
    batch_size = in_0.shape[0]
    in_channels = in_0.shape[1]
    in_height = in_0.shape[2]
    in_width = in_0.shape[3]
    
    out_channels = in_1.shape[1]
    out_height = in_1.shape[2]
    out_width = in_1.shape[3]
    
    # Verify that spatial dimensions match for concatenation
    assert out_height == in_1.shape[2] and out_width == in_1.shape[3], "Output spatial dimensions must match"
    
    # Create output tensor
    output_shape = (batch_size, in_channels + out_channels, out_height, out_width)
    out = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Configuration for grid and block sizes
    BLOCK_SIZE_M = 64  # Process 64 channels at a time
    BLOCK_SIZE_N = 32  # Process 32 spatial elements at a time
    
    # Calculate grid dimensions
    M = in_channels + out_channels  # Total output channels
    N = out_height * out_width      # Total spatial elements
    total_programs_M = int((M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M)
    total_programs_N = int((N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    grid = (total_programs_M, total_programs_N, batch_size)
    
    # Launch kernel
    fused_adaptive_pool2d_cat_kernel[grid](
        in_0,
        in_1,
        out,
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return fused_adaptive_pool2d_cat