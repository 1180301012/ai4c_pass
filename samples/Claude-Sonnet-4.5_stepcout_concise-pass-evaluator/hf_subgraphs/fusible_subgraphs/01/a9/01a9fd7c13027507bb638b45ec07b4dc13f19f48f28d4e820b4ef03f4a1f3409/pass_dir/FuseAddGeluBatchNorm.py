import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match the pattern: add + gelu + batch_norm (inference mode)
    """
    # Add operation (in_4 += in_5 in the model)
    add_result = in_4 + in_5
    
    # GELU activation
    gelu_out = torch.nn.functional.gelu(add_result, approximate='none')
    
    # Batch normalization (inference mode)
    bn_out = torch.nn.functional.batch_norm(gelu_out, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    
    # Return both outputs
    return gelu_out, bn_out


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    """Extract arguments for the replacement function"""
    return (in_4, in_5, in_0, in_1, in_3, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_add_gelu_bn_kernel(
    x_ptr,
    y_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    gelu_out_ptr,
    bn_out_ptr,
    n_elements,
    C,
    spatial_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: add + gelu + batch_norm
    Shape: [N, C, H, W] where spatial_size = H * W
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Step 1: Add
    add_result = x + y
    
    # Step 2: GELU
    # GELU formula: 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt_2 = 1.4142135623730951
    gelu_result = 0.5 * add_result * (1.0 + tl.math.erf(add_result / sqrt_2))
    
    # Step 3: Batch normalization
    # Determine which channel each element belongs to
    channel_idx = (offsets // spatial_size) % C
    
    # Load batch norm parameters for the corresponding channels
    running_mean = tl.load(running_mean_ptr + channel_idx, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr + channel_idx, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0)
    bias_val = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    
    # Batch norm formula: (x - mean) / sqrt(var + eps) * weight + bias
    std = tl.sqrt(running_var + eps)
    normalized = (gelu_result - running_mean) / std
    bn_result = normalized * weight + bias_val
    
    # Store outputs
    tl.store(gelu_out_ptr + offsets, gelu_result, mask=mask)
    tl.store(bn_out_ptr + offsets, bn_result, mask=mask)


@torch.fx.wrap
def fused_add_gelu_bn(x, y, running_mean, running_var, weight, bias):
    """
    Wrapper function for the fused kernel
    """
    # Get tensor dimensions
    N, C, H, W = x.shape
    spatial_size = H * W
    n_elements = x.numel()
    
    # Allocate output tensors
    gelu_out = torch.empty_like(x)
    bn_out = torch.empty_like(x)
    
    # Launch kernel
    eps = 1e-05
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_add_gelu_bn_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        gelu_out_ptr=gelu_out,
        bn_out_ptr=bn_out,
        n_elements=n_elements,
        C=C,
        spatial_size=spatial_size,
        eps=eps,
    )
    
    return gelu_out, bn_out


def replacement_func():
    """Return the replacement function (not called)"""
    return fused_add_gelu_bn