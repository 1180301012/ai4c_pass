import torch
import triton
import triton.language as tl

# Triton kernel for layer normalization (hidden_dim=768)
@triton.jit
def layernorm_kernel_768(
    X_ptr,
    Y_ptr,
    W_ptr,
    B_ptr,
    stride_x,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    X_row = X_ptr + row_idx * stride_x
    Y_row = Y_ptr + row_idx * stride_x
    
    # Load input - set out-of-bounds to 0 for correct reduction
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean - only sum valid elements
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = sum_x / N
    
    # Compute variance - only sum valid elements
    x_mean = tl.where(mask, x - mean, 0.0)
    sum_sq = tl.sum(x_mean * x_mean, axis=0)
    var = sum_sq / N
    
    # Normalize
    rstd = tl.rsqrt(var + eps)
    x_norm = (x - mean) * rstd
    
    # Apply affine transform
    w = tl.load(W_ptr + cols, mask=mask, other=1.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)
    y = x_norm * w + b
    
    # Store only valid elements
    tl.store(Y_row + cols, y, mask=mask)


@torch.fx.wrap
def triton_layernorm_768(x, weight, bias):
    """Triton layer norm for normalized_shape=768"""
    N = 768
    eps = 1e-05
    BLOCK_SIZE = 1024
    
    orig_shape = x.shape
    x_2d = x.reshape(-1, N).contiguous()
    M = x_2d.shape[0]
    
    y_2d = torch.empty_like(x_2d)
    
    grid = (M,)
    layernorm_kernel_768[grid](
        x_2d, y_2d, weight, bias,
        x_2d.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return y_2d.reshape(orig_shape)


def pattern(x, weight, bias):
    """Pattern: layer_norm(768) + dropout(0.0, training=False)"""
    y = torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-05)
    z = torch.nn.functional.dropout(y, 0.0, False, False)
    return z


def replacement_args(x, weight, bias):
    return (x, weight, bias)


def replacement_func():
    return triton_layernorm_768