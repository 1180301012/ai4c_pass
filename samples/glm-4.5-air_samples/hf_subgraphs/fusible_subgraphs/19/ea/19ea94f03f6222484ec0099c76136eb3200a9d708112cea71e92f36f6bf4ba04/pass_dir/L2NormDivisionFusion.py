import torch
import triton
import triton.language as tl

# Pattern matching function - matches L2 norm + division pattern
def pattern(in_1):
    norm_result = in_1.norm(p=2, dim=-1, keepdim=True)
    return norm_result

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# L2 Norm Kernel
@triton.jit
def l2_norm_kernel(
    in_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # Calculate L2 norm for this row
    sum_sq = 0.0
    for col_idx in tl.range(0, n_cols, BLOCK_SIZE):
        mask = col_idx < n_cols
        # Load input data
        input_data = tl.load(in_ptr + row_start + col_idx, mask=mask, other=0.0)
        # Sum of squares
        sum_sq += input_data * input_data
    
    # L2 norm (sqrt of sum of squares)
    l2_norm = tl.sqrt(sum_sq)
    
    # Store the norm result
    tl.store(out_ptr + row_idx, l2_norm)

@torch.fx.wrap
def l2_norm_division_fusion(in_1):
    # Input shape: [batch_size, features]
    batch_size, features = in_1.shape
    BLOCK_SIZE = 256
    
    # Create output tensor for norm results (shape: [batch_size, 1])
    out = torch.empty((batch_size, 1), dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel - this computes L2 norm
    num_programs = batch_size
    l2_norm_kernel[(num_programs,)](
        in_ptr=in_1,
        out_ptr=out,
        n_rows=batch_size,
        n_cols=features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return l2_norm_division_fusion