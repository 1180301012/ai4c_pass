import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation in model.py
def pattern(in_0, in_1):
    in_1 += in_0
    tmp_0 = in_1
    tmp_1 = tmp_0.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.type_as(tmp_0)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 8}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 16}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n_cols'],
)
@triton.jit
def fused_add_softmax_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (softmax is along last dim)
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load data from both inputs
    in_0_vals = tl.load(in_0_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
    in_1_vals = tl.load(in_1_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
    
    # Fused add
    x = in_0_vals + in_1_vals
    
    # Softmax - numerically stable version
    # 1. Find max for numerical stability
    x_max = tl.max(x, axis=0)
    # 2. Subtract max and compute exp
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)
    # 3. Sum of exp values
    sum_exp = tl.sum(exp_x, axis=0)
    # 4. Divide by sum to get softmax
    softmax_out = exp_x / sum_exp
    
    # Store result
    tl.store(out_ptr + row_start + col_offsets, softmax_out, mask=mask)

@torch.fx.wrap
def fused_add_softmax(in_0, in_1):
    # Get shape information
    shape = in_1.shape
    n_cols = shape[-1]
    n_rows = in_1.numel() // n_cols
    
    # Allocate output tensor with same shape and dtype
    out = torch.empty_like(in_1)
    
    # Launch kernel - one program per row
    grid = (n_rows,)
    fused_add_softmax_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    
    return out

# Replacement function - returns the optimized function
def replacement_func():
    return fused_add_softmax