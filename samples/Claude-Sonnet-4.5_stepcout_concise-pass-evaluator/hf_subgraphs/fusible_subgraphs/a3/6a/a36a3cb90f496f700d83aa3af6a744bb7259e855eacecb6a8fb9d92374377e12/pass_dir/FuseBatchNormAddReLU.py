import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """Match the entire computation: batch_norm + add + relu + mean"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = torch.nn.functional.batch_norm(in_4, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    return (tmp_6, tmp_7)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_bn_add_relu_kernel(
    input_ptr,
    residual_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    C,
    spatial_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for batch_norm + add + relu"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and residual
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    
    # Compute channel index
    channel_idx = (offsets // spatial_size) % C
    
    # Load batch norm parameters
    mean = tl.load(running_mean_ptr + channel_idx, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + channel_idx, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    
    # Batch norm: (x - mean) / sqrt(var + eps) * weight + bias
    normalized = (x - mean) / tl.sqrt(var + eps)
    bn_out = normalized * weight + bias
    
    # Add residual
    added = bn_out + residual
    
    # ReLU
    output = tl.maximum(added, 0.0)
    
    # Store
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 16}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 32, 'BLOCK_SIZE_W': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 16, 'BLOCK_SIZE_W': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_H': 8, 'BLOCK_SIZE_W': 32}, num_warps=2),
    ],
    key=['H', 'W'],
)
@triton.jit
def mean_spatial_kernel(
    input_ptr,
    output_ptr,
    N, C, H, W,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """Optimized kernel for spatial mean reduction"""
    pid_nc = tl.program_id(0)
    n = pid_nc // C
    c = pid_nc % C
    
    base_offset = n * C * H * W + c * H * W
    
    sum_val = 0.0
    
    for h_start in range(0, H, BLOCK_SIZE_H):
        for w_start in range(0, W, BLOCK_SIZE_W):
            h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
            w_offsets = w_start + tl.arange(0, BLOCK_SIZE_W)
            
            h_mask = h_offsets < H
            w_mask = w_offsets < W
            
            h_offsets_2d = h_offsets[:, None]
            w_offsets_2d = w_offsets[None, :]
            
            offsets = base_offset + h_offsets_2d * W + w_offsets_2d
            mask = h_mask[:, None] & w_mask[None, :]
            
            vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
            sum_val += tl.sum(vals)
    
    spatial_size = H * W
    mean_val = sum_val / spatial_size
    
    output_offset = n * C + c
    tl.store(output_ptr + output_offset, mean_val)

@torch.fx.wrap
def fused_bn_add_relu_mean(running_mean, running_var, bias, weight, input, residual):
    """Wrapper for fused batch_norm + add + relu + mean"""
    N, C, H, W = input.shape
    spatial_size = H * W
    n_elements = input.numel()
    eps = 1e-05
    
    # First compute bn + add + relu
    relu_output = torch.empty_like(input)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_bn_add_relu_kernel[grid](
        input,
        residual,
        running_mean,
        running_var,
        weight,
        bias,
        relu_output,
        n_elements,
        C,
        spatial_size,
        eps,
    )
    
    # Compute mean over spatial dimensions
    mean_output = torch.empty(N, C, 1, 1, dtype=input.dtype, device=input.device)
    
    mean_grid = (N * C,)
    mean_spatial_kernel[mean_grid](
        relu_output,
        mean_output,
        N, C, H, W,
    )
    
    return relu_output, mean_output

def replacement_func():
    return fused_bn_add_relu_mean