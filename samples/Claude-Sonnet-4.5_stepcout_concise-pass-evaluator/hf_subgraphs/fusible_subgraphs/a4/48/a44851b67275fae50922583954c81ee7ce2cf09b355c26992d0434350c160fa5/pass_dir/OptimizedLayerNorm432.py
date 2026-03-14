import torch
import triton
import triton.language as tl


def pattern(input_tensor, weight, bias):
    """Pattern: layer_norm with 432 dimensions"""
    output = torch.nn.functional.layer_norm(input_tensor, (432,), weight, bias, 1e-06)
    return output


def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized layer norm kernel - single pass with vectorized operations"""
    row_idx = tl.program_id(0)
    
    # Pointers to the current row
    row_start = row_idx * N
    
    # Load entire row at once (assuming N fits in BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load input
    x = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / N
    
    # Compute variance
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias_val = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Normalize and apply affine transformation
    out = x_centered * rstd * weight + bias_val
    
    # Store output
    tl.store(output_ptr + row_start + offsets, out, mask=mask)


@torch.fx.wrap
def optimized_layer_norm(input_tensor, weight, bias):
    """Wrapper for optimized layer norm kernel"""
    batch_size, seq_len, hidden_dim = input_tensor.shape
    M = batch_size * seq_len
    N = hidden_dim
    
    # Flatten batch and sequence dimensions
    input_flat = input_tensor.reshape(M, N).contiguous()
    
    # Allocate output tensor
    output_flat = torch.empty((M, N), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    grid = (M,)
    layer_norm_kernel[grid](
        input_flat,
        weight,
        bias,
        output_flat,
        M,
        N,
        1e-06,
    )
    
    # Reshape output
    output = output_flat.reshape(batch_size, seq_len, hidden_dim)
    
    return output


def replacement_func():
    return optimized_layer_norm