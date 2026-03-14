import torch
import triton
import triton.language as tl

# Pattern to match layer_norm with normalized_shape=(1024,)
def pattern(input_tensor, weight, bias):
    result = torch.nn.functional.layer_norm(input_tensor, (1024,), weight, bias, 1e-12)
    return result

def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)

# Triton kernel for layer_norm
@triton.jit
def layer_norm_fwd_kernel_1024(
    X_ptr,
    W_ptr,
    B_ptr,
    Y_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * N
    
    # Load input row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
    
    # Compute mean
    x_sum = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = x_sum / N
    
    # Compute variance
    x_centered = x - mean
    x_centered_masked = tl.where(mask, x_centered, 0.0)
    var_sum = tl.sum(x_centered_masked * x_centered_masked, axis=0)
    var = var_sum / N
    
    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd
    
    # Apply affine transform
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * w + b
    
    tl.store(Y_ptr + row_start + cols, y, mask=mask)


@torch.fx.wrap
def triton_layer_norm_1024(input_tensor, weight, bias):
    shape = input_tensor.shape
    N = shape[-1]
    M = input_tensor.numel() // N
    
    # Ensure contiguous and flatten
    x = input_tensor.contiguous().view(-1)
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 1024  # Fixed block size = 1024
    
    layer_norm_fwd_kernel_1024[(M,)](
        X_ptr=x,
        W_ptr=weight,
        B_ptr=bias,
        Y_ptr=output,
        N=N,
        eps=1e-12,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output.view(shape)

def replacement_func():
    return triton_layer_norm_1024