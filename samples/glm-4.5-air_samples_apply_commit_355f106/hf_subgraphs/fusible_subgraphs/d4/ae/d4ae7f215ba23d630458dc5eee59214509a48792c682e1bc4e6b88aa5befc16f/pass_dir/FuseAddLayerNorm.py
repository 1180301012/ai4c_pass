import torch
import triton
import triton.language as tl

def pattern(tmp_5, in_2, weight, bias, eps, hidden_size):
    tmp_6 = in_2 + tmp_5
    tmp_7 = torch.nn.functional.layer_norm(tmp_6, (hidden_size,), weight, bias, eps)
    return tmp_6, tmp_7

def replacement_args(tmp_5, in_2, weight, bias, eps, hidden_size):
    return (tmp_5, in_2, weight, bias, eps, hidden_size)

@triton.jit
def fused_add_layernorm_kernel(
    x1_ptr,
    x2_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    norm_output_ptr,
    N, H, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    has_weight: tl.constexpr,
    has_bias: tl.constexpr
):
    # Each program handles a block of positions
    row_start = tl.program_id(0) * BLOCK_SIZE
    col_start = tl.program_id(1) * BLOCK_SIZE
    row_offset = row_start + tl.arange(0, BLOCK_SIZE)
    col_offset = col_start + tl.arange(0, BLOCK_SIZE)
    
    mask1 = row_offset < N
    mask2 = col_offset < H
    
    # Load input tensor values
    x1 = tl.load(x1_ptr + row_offset * D + col_offset, mask=(mask1[:, None] & mask2[None, :]), other=0.0)
    x2 = tl.load(x2_ptr + row_offset * D + col_offset, mask=(mask1[:, None] & mask2[None, :]), other=0.0)
    
    # Addition operation
    x = x1 + x2
    
    # Layer normalization - compute mean and variance
    partial_mean = tl.sum(x, axis=1) / H
    partial_var = tl.sum((x - partial_mean[:, None]) ** 2, axis=1) / H
    
    # Compute normalized output
    mean = partial_mean
    var = partial_var + eps
    std = tl.sqrt(var)
    
    # Normalize
    x_norm = (x - mean[:, None]) / std[:, None]
    
    # Apply weight and bias if provided
    if has_weight:
        weight = tl.load(weight_ptr + col_offset, mask=mask2, other=1.0)
        x_norm = x_norm * weight
    
    if has_bias:
        bias = tl.load(bias_ptr + col_offset, mask=mask2, other=0.0)
        x_norm = x_norm + bias
    
    # Store results
    tl.store(output_ptr + row_offset * D + col_offset, x, mask=(mask1[:, None] & mask2[None, :]))
    tl.store(norm_output_ptr + row_offset * D + col_offset, x_norm, mask=(mask1[:, None] & mask2[None, :]))

@torch.fx.wrap
def fused_add_layernorm_ops(tmp_5, in_2, weight, bias, eps, hidden_size):
    # Reshape to 2D for processing
    tmp_5_2d = tmp_5.view(-1, hidden_size)
    in_2_2d = in_2.view(-1, hidden_size)
    
    N, H = tmp_5_2d.shape
    
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, BLOCK_SIZE), triton.cdiv(H, BLOCK_SIZE))
    
    output_2d = torch.empty_like(tmp_5_2d)
    norm_output_2d = torch.empty_like(tmp_5_2d)
    
    has_weight = weight is not None
    has_bias = bias is not None
    
    fused_add_layernorm_kernel[grid](
        x1_ptr=tmp_5_2d,
        x2_ptr=in_2_2d,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output_2d,
        norm_output_ptr=norm_output_2d,
        N=N, H=hidden_size, D=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
        has_weight=has_weight,
        has_bias=has_bias
    )
    
    return output_2d.view(1, -1, hidden_size), norm_output_2d.view(1, -1, hidden_size)

def replacement_func():
    return fused_add_layernorm_ops