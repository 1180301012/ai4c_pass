import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias):
    """Pattern to match: View + BatchNorm"""
    viewed = x.view(1, 512, 64, 64)
    bn_out = torch.nn.functional.batch_norm(viewed, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return bn_out

def replacement_args(x, running_mean, running_var, weight, bias):
    """Extract arguments for the replacement function"""
    return (x, running_mean, running_var, weight, bias)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def fused_batchnorm_kernel(
    x_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, out_ptr,
    N, C, H, W,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for BatchNorm in inference mode"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    n_elements = N * C * H * W
    mask = offsets < n_elements
    
    # Calculate channel index for each element
    # For layout [N, C, H, W], element at index i has channel (i // (H*W)) % C
    HW = H * W
    c_idx = (offsets // HW) % C
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load BatchNorm parameters (broadcast by channel)
    mean = tl.load(running_mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(running_var_ptr + c_idx, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    # BatchNorm computation: (x - mean) / sqrt(var + eps) * weight + bias
    normalized = (x - mean) / tl.sqrt(var + eps)
    scaled = normalized * w + b
    
    # Store result
    tl.store(out_ptr + offsets, scaled, mask=mask)

@torch.fx.wrap
def fused_view_bn(x, running_mean, running_var, weight, bias, eps=1e-5):
    """Wrapper function for the fused View + BatchNorm kernel"""
    # Apply view to reshape input
    viewed = x.view(1, 512, 64, 64)
    N, C, H, W = viewed.shape
    out = torch.empty_like(viewed)
    
    n_elements = N * C * H * W
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_batchnorm_kernel[grid](
        viewed, running_mean, running_var, weight, bias, out,
        N, C, H, W,
        eps=eps,
    )
    
    return out

def replacement_func():
    """Return the replacement function (not called)"""
    return fused_view_bn