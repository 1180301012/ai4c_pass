import torch
import triton
import triton.language as tl

# Pattern matching function: start with just Conv2D to test matching
def pattern(in_1, in_0):
    # Try just the convolution first
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
    return (tmp_1,)  # Return intermediate to match the original return structure

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_1, in_0)  # Return in pattern argument order: input, weight

# Simple optimized kernel for just Conv2D operation
@triton.jit
def conv_kernel(
    x_ptr,          # Input tensor [1, 256, 32, 32] -> flattened: [256*32*32]
    weight_ptr,     # Weight tensor [128, 256, 1, 1]  -> flattened: [128*256]
    out_ptr,        # Output tensor [1, 128, 32, 32] -> flattened: [128*32*32]
    n_channels_in,  # 256
    n_channels_out, # 128
    feat_h,         # 32
    feat_w,         # 32
    BLOCK_SIZE_M: tl.constexpr,    # Output channels per block
    BLOCK_SIZE: tl.constexpr,      # Linearized spatial locations per block
):
    # Each program handles a block of output channels and linearized spatial locations
    m_idx = tl.program_id(0)  # Output channel block
    n_idx = tl.program_id(1)  # Linear spatial location block
    
    # Output channel indices for this block
    m_start = m_idx * BLOCK_SIZE_M
    m_end = min(m_start + BLOCK_SIZE_M, n_channels_out)
    
    # Total spatial locations
    total_spatial = feat_h * feat_w
    
    # Process spatial locations in this block
    offsets = n_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_spatial
    
    linear_spatial = offsets[mask]
    
    # Process output channels in this block
    for m in range(m_start, m_end):
        for lin_idx in linear_spatial:
            # Convert linear index to (h, w)
            h_idx = lin_idx // feat_w
            w_idx = lin_idx % feat_w
            
            # Perform 1x1 convolution for this output channel and spatial location
            conv_result = 0.0
            for c_in in range(n_channels_in):
                # Weight for (m, c_in) - weight is [128, 256, 1, 1]
                weight_val = tl.load(weight_ptr + m * n_channels_in + c_in, mask=True).to(tl.float32)
                
                # Input value at (c_in, h_idx, w_idx) - input is [1, 256, 32, 32]
                in_offset = c_in * total_spatial + h_idx * feat_w + w_idx
                in_val = tl.load(x_ptr + in_offset, mask=True).to(tl.float32)
                
                conv_result += in_val * weight_val
            
            # Store result at output location [m, h_idx, w_idx]
            # Output is [1, 128, 32, 32] -> flattened as [128, 32*32] = [128, 1024]
            out_offset = m * total_spatial + h_idx * feat_w + w_idx
            tl.store(out_ptr + out_offset, conv_result, mask=True)

@torch.fx.wrap
def fused_conv_unfold_reshape(x, weight):
    # Input: x [1, 256, 32, 32], weight [128, 256, 1, 1]
    # Output: [1, 128, 32, 32] (just the convolution result)
    
    n_channels_in = x.shape[1]
    n_channels_out = weight.shape[0]
    feat_h = x.shape[2]
    feat_w = x.shape[3]
    
    # Output shape for convolution: [1, 128, 32, 32]
    out_shape = (1, n_channels_out, feat_h, feat_w)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Configure block sizes for optimal GPU occupancy
    BLOCK_SIZE_M = 32  # Output channels per block
    BLOCK_SIZE = 256  # Linearized spatial locations per block
    
    # Total spatial locations
    total_spatial = feat_h * feat_w
    
    # Calculate grid dimensions
    grid_m = (n_channels_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (total_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    conv_kernel[(grid_m, grid_n)](
        x_ptr=x,
        weight_ptr=weight,
        out_ptr=out,
        n_channels_in=n_channels_in,
        n_channels_out=n_channels_out,
        feat_h=feat_h,
        feat_w=feat_w,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_conv_unfold_reshape