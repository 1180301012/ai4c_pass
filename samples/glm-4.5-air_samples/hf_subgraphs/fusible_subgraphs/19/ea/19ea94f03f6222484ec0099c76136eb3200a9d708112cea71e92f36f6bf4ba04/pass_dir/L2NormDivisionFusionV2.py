import torch
import triton
import triton.language as tl

# Pattern matching function - matches the complete normalization pattern
def pattern(in_1):
    # This matches the sequence: norm followed by division
    norm_result = in_1.norm(p=2, dim=-1, keepdim=True)
    normalized_result = in_1 / norm_result
    # Return both to match the graph structure - norm_result is used and then set to None
    return norm_result, normalized_result

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Fused L2 Norm + Division Kernel
@triton.jit
def fused_norm_div_kernel(
    in_ptr,
    out_ptr,
    norm_ptr,
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
    
    # Prevent division by zero
    l2_norm = tl.maximum(l2_norm, 1e-6)
    
    # Store the norm result (needed for graph consistency)
    tl.store(norm_ptr + row_idx, l2_norm)
    
    # Normalize all elements in the row
    for col_idx in tl.range(0, n_cols, BLOCK_SIZE):
        mask = col_idx < n_cols
        # Load input data
        input_data = tl.load(in_ptr + row_start + col_idx, mask=mask, other=0.0)
        # Normalize and store
        normalized_data = input_data / l2_norm
        tl.store(out_ptr + row_start + col_idx, normalized_data, mask=mask)

@torch.fx.wrap
def fused_norm_div_fusion(in_1):
    # Input shape: [batch_size, features]
    batch_size, features = in_1.shape
    
    # Optimized block size for this workload
    BLOCK_SIZE = 128  # Smaller block size for better memory access pattern
    
    # Create output tensor for normalized result
    out = torch.empty_like(in_1)
    
    # Create tensor to store norm results (shape: [batch_size, 1])
    norm_result = torch.empty((batch_size, 1), dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel - computes both norm and normalized result in one pass
    num_programs = batch_size
    fused_norm_div_kernel[(num_programs,)](
        in_ptr=in_1,
        out_ptr=out,
        norm_ptr=norm_result,
        n_rows=batch_size,
        n_cols=features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return norm_result, out

# Replacement function
def replacement_func():
    return fused_norm_div_fusion