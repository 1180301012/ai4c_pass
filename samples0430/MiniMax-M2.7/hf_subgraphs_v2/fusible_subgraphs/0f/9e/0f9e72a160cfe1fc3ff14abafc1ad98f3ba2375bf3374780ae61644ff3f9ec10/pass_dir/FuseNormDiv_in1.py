import torch
import triton
import triton.language as tl

@triton.jit
def normalize_l2_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused normalize-divide kernel for L2 normalization along last dimension.
    Uses a single program to load all elements for the reduction."""
    # Use single program (pid=0) for reduction operation
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Cast to fp32 for math operations (rsqrt doesn't support fp16/bf16)
    x_fp32 = x.to(tl.float32)
    
    # Compute sum of squares for reduction
    x_sq = x_fp32 * x_fp32
    sum_sq = tl.sum(x_sq, axis=0)
    
    # Compute 1/sqrt(sum(x^2)) for normalization
    rsqrt = tl.math.rsqrt(sum_sq + 1e-6)
    
    # Normalize: x / norm = x * rsqrt (broadcast rsqrt to all elements)
    x_norm = x * rsqrt
    
    tl.store(out_ptr + offsets, x_norm, mask=mask)

def pattern(in_1):
    tmp_1 = in_1.norm(p = 2, dim = -1, keepdim = True)
    tmp_2 = in_1 / tmp_1
    return tmp_2

def replacement_args(in_1):
    return (in_1,)

@torch.fx.wrap
def normalize_l2_wrapper(in_1):
    N = in_1.numel()
    # Use BLOCK_SIZE >= N to ensure single program for proper reduction
    BLOCK_SIZE = N  # 512
    
    out = torch.empty_like(in_1)
    
    normalize_l2_kernel[(1,)](
        x_ptr=in_1,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return normalize_l2_wrapper