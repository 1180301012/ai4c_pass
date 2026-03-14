import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Pattern: flatten(2) followed by transpose(1, 2)
    Input shape: [N, C, D, H, W]
    After flatten(2): [N, C, D*H*W]
    After transpose(1, 2): [N, D*H*W, C]
    """
    tmp_4 = x.flatten(2)
    tmp_5 = tmp_4.transpose(1, 2)
    return tmp_5

def replacement_args(x):
    return (x,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 256, 'BLOCK_S': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 256, 'BLOCK_S': 64}, num_warps=8),
        triton.Config({'BLOCK_C': 512, 'BLOCK_S': 32}, num_warps=8),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 128}, num_warps=8),
        triton.Config({'BLOCK_C': 64, 'BLOCK_S': 64}, num_warps=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_S': 128}, num_warps=4),
    ],
    key=['C', 'spatial_size'],
)
@triton.jit
def flatten_transpose_kernel(
    input_ptr,
    output_ptr,
    N, C, spatial_size,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """
    Optimized fused kernel for flatten(2) + transpose(1, 2)
    Input: [N, C, spatial_size] (already conceptually flattened)
    Output: [N, spatial_size, C]
    
    2D tiling strategy for better performance
    """
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_tile = tl.program_id(1)
    
    # Calculate which tile we're processing
    num_c_tiles = tl.cdiv(C, BLOCK_C)
    pid_c = pid_tile % num_c_tiles
    pid_s = pid_tile // num_c_tiles
    
    # Calculate offsets
    c_start = pid_c * BLOCK_C
    s_start = pid_s * BLOCK_S
    
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    
    c_mask = c_offsets < C
    s_mask = s_offsets < spatial_size
    
    # Input layout: [N, C, spatial_size]
    # For each channel in our block, load all spatial positions in our block
    for s_idx in range(BLOCK_S):
        s = s_start + s_idx
        if s < spatial_size:
            # Input index: batch * C * spatial_size + c * spatial_size + s
            input_idx = pid_batch * C * spatial_size + c_offsets * spatial_size + s
            
            # Load from input
            vals = tl.load(input_ptr + input_idx, mask=c_mask, other=0.0)
            
            # Output layout: [N, spatial_size, C]
            # Output index: batch * spatial_size * C + s * C + c
            output_idx = pid_batch * spatial_size * C + s * C + c_offsets
            
            # Store to output
            tl.store(output_ptr + output_idx, vals, mask=c_mask)

@torch.fx.wrap
def fused_flatten_transpose(x):
    """
    Fused implementation of flatten(2) + transpose(1, 2)
    """
    # Get input shape
    shape = x.shape
    N = shape[0]
    C = shape[1]
    
    # Calculate spatial size (product of all dimensions after dim 1)
    spatial_size = 1
    for i in range(2, len(shape)):
        spatial_size *= shape[i]
    
    # Create output tensor
    output = torch.empty((N, spatial_size, C), dtype=x.dtype, device=x.device)
    
    # Launch kernel - autotune will find the best BLOCK_C and BLOCK_S
    # Grid size will be adjusted by the kernel based on chosen block sizes
    grid = lambda meta: (N, triton.cdiv(C, meta['BLOCK_C']) * triton.cdiv(spatial_size, meta['BLOCK_S']))
    
    flatten_transpose_kernel[grid](
        x,
        output,
        N, C, spatial_size,
    )
    
    return output

def replacement_func():
    return fused_flatten_transpose