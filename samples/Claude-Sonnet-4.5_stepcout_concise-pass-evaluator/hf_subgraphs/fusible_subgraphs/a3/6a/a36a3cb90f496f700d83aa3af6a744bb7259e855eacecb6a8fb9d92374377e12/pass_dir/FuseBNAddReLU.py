import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """Match batch_norm + add pattern"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = torch.nn.functional.batch_norm(in_4, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    return tmp_5

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_4, in_0, in_1, in_3, in_2, in_5)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_SPATIAL': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_SPATIAL': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE_SPATIAL': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_SPATIAL': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE_SPATIAL': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE_SPATIAL': 2048}, num_warps=8),
    ],
    key=['spatial_size'],
)
@triton.jit
def fused_bn_add_kernel(
    input_ptr,
    residual_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N, C, spatial_size,
    eps: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    """Fused kernel for batch_norm + add - optimized for coalesced access"""
    # Each program handles one spatial block across one (N, C) pair
    pid_nc = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    
    n = pid_nc // C
    c = pid_nc % C
    
    # Load batch norm parameters for this channel (scalar loads)
    mean = tl.load(running_mean_ptr + c)
    var = tl.load(running_var_ptr + c)
    weight = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)
    
    # Precompute normalization factor
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale = weight * inv_std
    shift = bias - mean * scale
    
    # Compute base offset and spatial offsets
    base_offset = n * C * spatial_size + c * spatial_size
    spatial_start = pid_spatial * BLOCK_SIZE_SPATIAL
    offsets = spatial_start + tl.arange(0, BLOCK_SIZE_SPATIAL)
    mask = offsets < spatial_size
    
    # Load input and residual
    x = tl.load(input_ptr + base_offset + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + base_offset + offsets, mask=mask, other=0.0)
    
    # Fused: batch_norm + add
    output = x * scale + shift + residual
    
    # Store
    tl.store(output_ptr + base_offset + offsets, output, mask=mask)

@torch.fx.wrap
def fused_bn_add(input, running_mean, running_var, weight, bias, residual):
    """Wrapper for fused batch_norm + add"""
    N, C, H, W = input.shape
    spatial_size = H * W
    eps = 1e-05
    
    output = torch.empty_like(input)
    
    # 2D grid: (N * C, num_spatial_blocks)
    BLOCK_SIZE_SPATIAL = 256  # Will be tuned by autotune
    num_spatial_blocks = triton.cdiv(spatial_size, BLOCK_SIZE_SPATIAL)
    grid = (N * C, num_spatial_blocks)
    
    fused_bn_add_kernel[grid](
        input,
        residual,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        N, C, spatial_size,
        eps,
    )
    
    return output

def replacement_func():
    return fused_bn_add