import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Match the pattern: ReLU (inplace) + Dropout2d (eval mode)
    """
    tmp_0 = torch.nn.functional.relu(x, inplace=True)
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return (tmp_1, tmp_0)

def replacement_args(x):
    """
    Extract arguments for replacement
    """
    return (x,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized ReLU kernel using Triton
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(x, 0)
    out = tl.maximum(x, 0.0)
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_relu_dropout(x):
    """
    Fused ReLU + Dropout2d (eval mode)
    Since dropout in eval mode is identity, we only compute ReLU
    """
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Calculate grid size
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    relu_kernel[grid](x, out, N)
    
    # Return the same output twice to match original pattern
    # (tmp_1, tmp_0) where tmp_1 = dropout(tmp_0) = tmp_0 in eval mode
    return (out, out)

def replacement_func():
    """
    Return the replacement function
    """
    return optimized_relu_dropout