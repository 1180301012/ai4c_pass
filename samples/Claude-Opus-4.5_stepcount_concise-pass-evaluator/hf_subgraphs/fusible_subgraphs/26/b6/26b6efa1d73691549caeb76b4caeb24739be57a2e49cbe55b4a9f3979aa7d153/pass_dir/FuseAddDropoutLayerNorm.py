import torch
import triton
import triton.language as tl

# Pattern matching for layer_norm with normalized_shape=(64,)
def pattern(bias, weight, x):
    return torch.nn.functional.layer_norm(x, (64,), weight, bias, 1e-12)

def replacement_args(bias, weight, x):
    return (bias, weight, x)

@triton.jit
def layer_norm_kernel_64(
    X,  # input pointer [M, 64]
    Y,  # output pointer [M, 64]
    W,  # weight pointer [64]
    B,  # bias pointer [64]
    M,  # number of rows
    BLOCK_SIZE: tl.constexpr,  # 64
):
    # One program per row
    row = tl.program_id(0)
    
    # Load the entire row (64 elements)
    offs = tl.arange(0, BLOCK_SIZE)
    x_ptr = X + row * BLOCK_SIZE + offs
    x = tl.load(x_ptr).to(tl.float32)
    
    # Compute mean
    mean = tl.sum(x, axis=0) / BLOCK_SIZE
    
    # Compute variance  
    x_zm = x - mean
    var = tl.sum(x_zm * x_zm, axis=0) / BLOCK_SIZE
    
    # Compute rstd
    rstd = 1.0 / tl.sqrt(var + 1e-12)
    
    # Normalize
    x_hat = x_zm * rstd
    
    # Load weight and bias
    w = tl.load(W + offs).to(tl.float32)
    b = tl.load(B + offs).to(tl.float32)
    
    # Apply affine transformation
    y = x_hat * w + b
    
    # Store result
    y_ptr = Y + row * BLOCK_SIZE + offs
    tl.store(y_ptr, y)


@torch.fx.wrap
def triton_layer_norm(bias, weight, x):
    """
    Layer norm for hidden_size=64
    """
    orig_shape = x.shape
    H = 64  # Fixed for this pattern
    
    # Reshape to 2D
    x_2d = x.contiguous().view(-1, H)
    M = x_2d.shape[0]
    
    # Allocate output
    y = torch.empty_like(x_2d)
    
    # Launch kernel
    layer_norm_kernel_64[(M,)](
        x_2d, y, weight, bias,
        M,
        BLOCK_SIZE=64,
    )
    
    return y.view(orig_shape)


def replacement_func():
    return triton_layer_norm