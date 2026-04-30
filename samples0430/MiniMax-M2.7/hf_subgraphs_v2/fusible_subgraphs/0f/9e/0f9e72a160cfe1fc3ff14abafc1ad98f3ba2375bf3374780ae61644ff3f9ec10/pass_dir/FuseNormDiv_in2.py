import torch
import triton
import triton.language as tl

@triton.jit
def normalize_l2_3d_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused normalize-divide kernel for L2 normalization on [1,1,512] tensor."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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

def pattern(in_2):
    tmp_3 = in_2.norm(p = 2, dim = -1, keepdim = True)
    tmp_4 = in_2 / tmp_3
    return tmp_4

def replacement_args(in_2):
    return (in_2,)

@torch.fx.wrap
def normalize_l2_3d_wrapper(in_2):
    N = in_2.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_2)
    
    normalize_l2_3d_kernel[(num_programs,)](
        x_ptr=in_2,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return normalize_l2_3d_wrapper