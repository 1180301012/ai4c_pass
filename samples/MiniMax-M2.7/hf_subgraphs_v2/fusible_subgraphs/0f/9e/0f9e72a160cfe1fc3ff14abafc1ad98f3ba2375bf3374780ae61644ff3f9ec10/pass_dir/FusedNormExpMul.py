import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Match the pattern:
    tmp_1 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_2 = in_1 / tmp_1
    tmp_3 = in_2.norm(p=2, dim=-1, keepdim=True)
    tmp_4 = in_2 / tmp_3
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * tmp_4
    return (tmp_6, tmp_4, tmp_2)
    """
    tmp_1 = in_1.norm(p=2, dim=-1, keepdim=True)
    tmp_2 = in_1 / tmp_1
    tmp_3 = in_2.norm(p=2, dim=-1, keepdim=True)
    tmp_4 = in_2 / tmp_3
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * tmp_4
    return (tmp_6, tmp_4, tmp_2)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_norm_exp_mul_kernel(
    in_1_ptr, in_2_ptr, exp_val,
    out_tmp2_ptr, out_tmp4_ptr, out_tmp6_ptr,
    n_in_1_rows, n_in_1_cols,
    n_in_2_rows, n_in_2_cols, n_in_2_depth,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for:
    1. L2 normalize in_1 -> tmp_2
    2. L2 normalize in_2 -> tmp_4  
    3. tmp_6 = exp(in_0) * tmp_4
    
    in_1 shape: [1, 512]
    in_2 shape: [1, 1, 512]
    """
    # Program 0 handles in_1 normalization (shape [1, 512])
    pid = tl.program_id(0)
    
    # Handle in_1 normalization - compute L2 norm across dim=-1
    # For shape [1, 512] with dim=-1, we reduce across the 512 elements
    if pid == 0:
        # Compute sum of squares for in_1
        sum_sq = 0.0
        for i in range(n_in_1_cols):
            offset = i
            val = tl.load(in_1_ptr + offset)
            sum_sq += val * val
        norm_1 = tl.sqrt(sum_sq + 1e-8)
        
        # Normalize and store tmp_2
        for i in range(n_in_1_cols):
            offset = i
            val = tl.load(in_1_ptr + offset)
            normalized = val / norm_1
            tl.store(out_tmp2_ptr + offset, normalized)
    
    # Programs 1+ handle in_2 normalization
    # For in_2 shape [1, 1, 512], dim=-1 reduces across 512 elements
    # with keepdim=True, output shape is [1, 1, 1]
    pid_2 = pid - 1
    if pid_2 >= 0 and pid_2 < n_in_2_rows * n_in_2_cols:
        row_idx = pid_2 // n_in_2_cols
        col_idx = pid_2 % n_in_2_cols
        
        # Compute L2 norm for this slice
        base_offset = row_idx * n_in_2_cols * n_in_2_depth + col_idx * n_in_2_depth
        sum_sq_2 = 0.0
        for d in range(n_in_2_depth):
            val = tl.load(in_2_ptr + base_offset + d)
            sum_sq_2 += val * val
        norm_2 = tl.sqrt(sum_sq_2 + 1e-8)
        
        # Normalize and store tmp_4, compute tmp_6
        for d in range(n_in_2_depth):
            offset = base_offset + d
            val = tl.load(in_2_ptr + offset)
            normalized = val / norm_2
            tl.store(out_tmp4_ptr + offset, normalized)
            # tmp_6 = exp_val * normalized
            tmp_6_val = exp_val * normalized
            tl.store(out_tmp6_ptr + offset, tmp_6_val)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2):
    """
    Wrapper for the fused kernel that handles:
    1. L2 normalize in_1 -> tmp_2
    2. L2 normalize in_2 -> tmp_4
    3. tmp_6 = exp(in_0) * tmp_4
    """
    # Compute exp(in_0) using torch
    exp_val = torch.exp(in_0).item()
    
    n_in_1_rows, n_in_1_cols = in_1.shape
    n_in_2_rows, n_in_2_cols, n_in_2_depth = in_2.shape
    
    # Allocate outputs
    tmp_2 = torch.empty_like(in_1)
    tmp_4 = torch.empty_like(in_2)
    tmp_6 = torch.empty_like(in_2)
    
    # Grid: 1 for in_1 normalization + n_in_2_rows * n_in_2_cols for in_2
    grid = (1 + n_in_2_rows * n_in_2_cols,)
    BLOCK_SIZE = 512
    
    fused_norm_exp_mul_kernel[grid](
        in_1, in_2, exp_val,
        tmp_2, tmp_4, tmp_6,
        n_in_1_rows, n_in_1_cols,
        n_in_2_rows, n_in_2_cols, n_in_2_depth,
        BLOCK_SIZE,
    )
    
    return tmp_6, tmp_4, tmp_2


def replacement_func():
    return fused_kernel_wrapper