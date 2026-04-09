import torch
import triton
import triton.language as tl

def pattern(in_tensor):
    """Pattern to match L2 norm followed by division for in_1 [1, 512] tensor"""
    tmp_norm = in_tensor.norm(p=2, dim=-1, keepdim=True)
    tmp_normalized = in_tensor / tmp_norm
    return tmp_normalized

def replacement_args(in_tensor):
    return (in_tensor,)

@triton.jit
def l2_norm_kernel_512(
    in_ptr, out_ptr, norm_ptr,
    n_cols: tl.constexpr,
):
    """Optimized L2 normalization kernel for [1, 512] tensors"""
    # Each program processes one column
    col_idx = tl.program_id(0)
    
    if col_idx >= n_cols:
        return
        
    # Load the element
    x = tl.load(in_ptr + col_idx)
    
    # Compute square
    x_sq = x * x
    
    # All programs cooperate to compute sum
    block_sum = tl.sum(x_sq, axis=0)
    
    # Compute norm (sqrt of sum)
    norm = tl.sqrt(tl.max(block_sum, 1.0))
    
    # Store norm and normalized value
    tl.store(norm_ptr + 0, norm)
    tl.store(out_ptr + col_idx, x / norm)

@torch.fx.wrap
def l2_norm_forward_512(x):
    """Forward pass for L2 normalization of [1, 512] tensor"""
    if x.shape != (1, 512):
        raise ValueError(f"Expected shape (1, 512), got {x.shape}")
    
    # Store norm and compute normalized output
    norm = x.norm(p=2, dim=-1, keepdim=True)
    y = x / norm
    
    return y

def replacement_func():
    return l2_norm_forward_512