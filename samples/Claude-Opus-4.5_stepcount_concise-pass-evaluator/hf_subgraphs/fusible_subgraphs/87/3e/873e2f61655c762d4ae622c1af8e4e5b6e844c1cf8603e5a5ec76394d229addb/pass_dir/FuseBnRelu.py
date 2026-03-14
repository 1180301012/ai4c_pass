import torch
import triton
import triton.language as tl

# Pattern matching function - match BatchNorm + ReLU
def pattern(x, running_mean, running_var, weight, bias):
    """
    Match BatchNorm + ReLU pattern:
    - tmp_9 = batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    - tmp_10 = relu(tmp_9, inplace=True)
    """
    bn_result = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    relu_result = torch.nn.functional.relu(bn_result, inplace=True)
    return relu_result

# Argument extraction function
def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr,
    running_mean_ptr, running_var_ptr,
    weight_ptr, bias_ptr,
    out_ptr,
    total_elements,
    C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused BatchNorm (inference) + ReLU kernel
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < total_elements
    
    # Calculate channel index for each element
    # Layout is NCHW, so channel = (offset // HW) % C
    c_idx = (offsets // HW) % C
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load per-channel params
    running_mean = tl.load(running_mean_ptr + c_idx, mask=mask, other=0.0)
    running_var = tl.load(running_var_ptr + c_idx, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    # BatchNorm (inference mode)
    inv_std = 1.0 / tl.sqrt(running_var + eps)
    normalized = (x - running_mean) * inv_std
    scaled = weight * normalized + bias
    
    # ReLU
    out = tl.maximum(scaled, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def bn_relu_fused(x, running_mean, running_var, weight, bias):
    """
    Fused kernel wrapper for BatchNorm + ReLU
    """
    N, C, H, W = x.shape
    total_elements = N * C * H * W
    HW = H * W
    
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    bn_relu_kernel[grid](
        x,
        running_mean, running_var,
        weight, bias,
        out,
        total_elements,
        C, HW,
        eps=1e-05,
    )
    
    return out

def replacement_func():
    return bn_relu_fused