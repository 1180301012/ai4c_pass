import torch
import triton
import triton.language as tl

# Pattern matching for permute + contiguous fusion
def pattern(input_tensor):
    """
    Match the computation pattern:
    tmp_6 = input_tensor.view(64, 64, -1)
    tmp_7 = tmp_6.permute(2, 0, 1)
    tmp_8 = tmp_7.contiguous()
    """
    tmp_6 = input_tensor.view(64, 64, -1)
    tmp_7 = tmp_6.permute(2, 0, 1)
    tmp_8 = tmp_7.contiguous()
    return tmp_8

# Extract arguments for the replacement
def replacement_args(input_tensor):
    return (input_tensor,)

# Optimized Triton kernel for view + permute + contiguous fusion
@triton.jit
def view_permute_contiguous_kernel(
    in_ptr,          # Input tensor [height, width, n_features] = [64, 64, n]
    out_ptr,         # Output tensor [n_features, height, width] = [n, 64, 64]
    n_features,      # n (the feature dimension)
    height,          # 64 
    width,           # 64
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles a [BLOCK_SIZE_M, BLOCK_SIZE_N] tile
    m_pid = tl.program_id(0)
    n_pid = tl.program_id(1)
    f_pid = tl.program_id(2)
    
    # Compute tile bounds
    h_start = m_pid * BLOCK_SIZE_M
    w_start = n_pid * BLOCK_SIZE_N
    f = f_pid
    
    # Compute bounds
    mask_h = h_start + tl.arange(0, BLOCK_SIZE_M) < height
    mask_w = w_start + tl.arange(0, BLOCK_SIZE_N) < width
    
    # Load tile from input: [h, w, f] -> transpose to [f, h, w]
    # Original layout: height * width * n_features stride
    # Target layout: n_features * height * width stride
    
    offsets_h = h_start + tl.arange(0, BLOCK_SIZE_M)
    offsets_w = w_start + tl.arange(0, BLOCK_SIZE_N)
    
    # Load tile data for current feature f
    for i in range(BLOCK_SIZE_M):
        if (h_start + i) < height:
            for j in range(BLOCK_SIZE_N):
                if (w_start + j) < width:
                    # Source: [h_start+i, w_start+j, f] in input tensor
                    src_offset = (h_start + i) * width * n_features + (w_start + j) * n_features + f
                    # Target: [f, h_start+i, w_start+j] in output tensor  
                    dst_offset = f * height * width + (h_start + i) * width + (w_start + j)
                    
                    # Load and store directly
                    value = tl.load(in_ptr + src_offset)
                    tl.store(out_ptr + dst_offset, value)

# Optimized wrapper function
@torch.fx.wrap
def optimized_view_permute_contiguous(input_tensor):
    """Fused view + permute + contiguous operation"""
    
    # Get tensor properties
    n_input = input_tensor.shape[-1]  # The smallest dimension after view
    height = 64
    width = 64
    
    # Output shape should be [n_input, height, width]
    output = torch.empty((n_input, height, width), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel with 3D grid
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 8
    grid_m = triton.cdiv(height, BLOCK_SIZE_M)
    grid_n = triton.cdiv(width, BLOCK_SIZE_N)
    grid_dims = (grid_m, grid_n, n_input)
    
    view_permute_contiguous_kernel[grid_dims](
        input_tensor,
        output,
        n_input,
        height,
        width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_view_permute_contiguous