import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x, mean, var, weight, bias, affine, momentum, eps):
    """Match the batch_norm operation with affine=False"""
    if affine:
        return None
    return torch.nn.functional.batch_norm(x, mean, var, weight, bias, affine, momentum, eps)

# Argument extraction function
def replacement_args(x, mean, var, weight, bias, affine, momentum, eps):
    """Extract arguments for the replacement"""
    return (x, mean, var, momentum, eps)

# Triton kernel for normalized batch_norm (affine=False)
@triton.jit
def batch_norm_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    out_ptr,
    n_channels,
    n_batch,
    n_height,
    n_width,
    eps: tl.constexpr,
):
    c = tl.program_id(0)
    
    start = c * (n_batch * n_height * n_width)
    n_pixels = n_batch * n_height * n_width
    
    mean_val = tl.load(mean_ptr + c)
    var_val = tl.load(var_ptr + c)
    
    for i in range(0, n_pixels, 1024):
        x = tl.load(x_ptr + start + i, mask=i < n_pixels, other=0.0)
        normalized_x = (x - mean_val) / tl.sqrt(var_val + eps)
        tl.store(out_ptr + start + i, normalized_x, mask=i < n_pixels)

# Wrapper function
@torch.fx.wrap
def batch_norm(x, mean, var, momentum, eps):
    # Transpose to [C, B, H, W] for contiguous channel access
    x = x.permute(1, 0, 2, 3)
    
    # Ensure mean and var are float32
    mean = mean.to(torch.float32)
    var = var.to(torch.float32)
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Get shape information
    n_channels = x.shape[0]
    n_batch = x.shape[1]
    n_height = x.shape[2]
    n_width = x.shape[3]
    
    # Launch kernel
    batch_norm_kernel[(n_channels,)](
        x_ptr=x,
        mean_ptr=mean,
        var_ptr=var,
        out_ptr=out,
        n_channels=n_channels,
        n_batch=n_batch,
        n_height=n_height,
        n_width=n_width,
        eps=eps,
    )
    
    # Transpose back to [B, C, H, W]
    return out.permute(1, 0, 2, 3)

# Replacement function
def replacement_func():
    return batch_norm