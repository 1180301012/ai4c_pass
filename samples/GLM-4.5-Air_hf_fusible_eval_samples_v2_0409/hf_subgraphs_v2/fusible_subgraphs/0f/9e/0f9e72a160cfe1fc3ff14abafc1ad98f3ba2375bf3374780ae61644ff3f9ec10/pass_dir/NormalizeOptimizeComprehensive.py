import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern to match the entire computation graph: normalization + exp + multiply"""
    # Normalize in_1 [1, 512]
    tmp_1 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_2 = in_1 / tmp_1
    
    # Normalize in_2 [1, 1, 512] 
    tmp_3 = in_2.norm(p=2, dim=-1, keepdim=True)
    tmp_4 = in_2 / tmp_3
    
    # Exponential and multiplication
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * tmp_4
    
    return (tmp_6, tmp_4, tmp_2)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

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

@triton.jit  
def l2_norm_kernel_1_1_512(
    in_ptr, out_ptr, norm_ptr,
    n_cols: tl.constexpr,
):
    """Optimized L2 normalization kernel for [1, 1, 512] tensors"""
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
def comprehensive_forward(in_0, in_1, in_2):
    """Comprehensive forward pass for the entire computation"""
    # Normalize in_1 [1, 512]
    norm_1 = in_1.norm(p=2, dim=-1, keepdim=True)
    normalized_1 = in_1 / norm_1
    
    # Normalize in_2 [1, 1, 512]
    norm_2 = in_2.norm(p=2, dim=-1, keepdim=True)
    normalized_2 = in_2 / norm_2
    
    # Exponential and multiplication
    exp_result = in_0.exp()
    mul_result = exp_result * normalized_2
    
    return (mul_result, normalized_2, normalized_1)

def replacement_func():
    return comprehensive_forward