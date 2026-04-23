import torch
import triton
import triton.language as tl

# Pattern matching function - matches the full RMSNorm + residual scale pattern
def pattern(in_0, in_1, in_2):
    tmp_2 = in_0 * in_2
    tmp_4 = tmp_2.float()
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_4 * tmp_8
    tmp_10 = in_1.float()
    tmp_11 = 1.0 + tmp_10
    tmp_12 = tmp_9 * tmp_11
    tmp_13 = tmp_12.type_as(tmp_2)
    return tmp_2, tmp_13

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Kernel 1: Compute squared values for RMSNorm
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def rmsnorm_squares_kernel(
    in_0_ptr,
    scalar,
    out_squared_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    x_scaled = x * scalar
    x_sq = x_scaled * x_scaled
    tl.store(out_squared_ptr + offsets, x_sq, mask=mask)

# Kernel 2: Compute normalization factor (sum of squares) and apply RMSNorm + weight scaling
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def rmsnorm_normalize_kernel(
    in_0_ptr,
    squared_ptr,
    weight_ptr,
    out_tmp2_ptr,
    out_tmp13_ptr,
    n_elements,
    n_features,
    scalar,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr = 1e-06,
):
    """
    Each program processes one row (n_features elements).
    Each row has shape [2048] and needs its own normalization factor.
    """
    row_id = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_features
    
    # Global offset within the full tensor
    global_offsets = row_id * n_features + offsets
    
    # Load scaled values and squared values
    x_scaled = tl.load(in_0_ptr + global_offsets, mask=mask, other=0.0)
    x_sq = tl.load(squared_ptr + global_offsets, mask=mask, other=0.0)
    
    # Sum squares for this row
    sum_sq = tl.sum(x_sq, axis=0)
    
    # Compute RMS and normalize
    rms = tl.sqrt(sum_sq / n_features + EPS)
    normalized = x_scaled / rms
    
    # Load weight and apply scaling
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    w_scaled = 1.0 + w
    out = normalized * w_scaled
    
    # Store results
    tl.store(out_tmp2_ptr + global_offsets, x_scaled, mask=mask)
    tl.store(out_tmp13_ptr + global_offsets, out, mask=mask)

@torch.fx.wrap
def fused_rmsnorm_residual_scale(in_0, in_1, in_2):
    """
    Fused kernel for:
    - Scalar multiplication: in_0 * in_2
    - RMSNorm: float -> pow(2) -> mean -> add_eps -> rsqrt -> multiply
    - Residual scaling: in_1 + 1 -> multiply with normalized
    
    Input shapes:
    - in_0: [1, 3, 2048] bfloat16
    - in_1: [2048] bfloat16  
    - in_2: scalar bfloat16
    """
    batch_size, n_heads, n_features = in_0.shape
    n_elements = in_0.numel()
    n_rows = batch_size * n_heads
    
    # Extract scalar value
    scalar = float(in_2.item()) if in_2.numel() == 1 else float(in_2)
    
    # Allocate intermediate and output tensors
    out_squared = torch.empty_like(in_0, dtype=torch.float32)
    out_tmp2 = torch.empty_like(in_0, dtype=torch.float32)
    out_tmp13 = torch.empty_like(in_0, dtype=torch.float32)
    
    # Kernel 1: Compute squared values
    grid_1 = (triton.cdiv(n_elements, 512),)
    rmsnorm_squares_kernel[grid_1](
        in_0_ptr=in_0.float(),
        scalar=scalar,
        out_squared_ptr=out_squared,
        n_elements=n_elements,
    )
    
    # Kernel 2: Normalize and scale
    # Grid: (n_rows, blocks_per_row)
    grid_2 = (n_rows, triton.cdiv(n_features, 512))
    rmsnorm_normalize_kernel[grid_2](
        in_0_ptr=in_0.float(),
        squared_ptr=out_squared,
        weight_ptr=in_1.float(),
        out_tmp2_ptr=out_tmp2,
        out_tmp13_ptr=out_tmp13,
        n_elements=n_elements,
        n_features=n_features,
        scalar=scalar,
    )
    
    # Convert outputs back to original dtype
    return out_tmp2.to(in_0.dtype), out_tmp13.to(in_0.dtype)

def replacement_func():
    return fused_rmsnorm_residual_scale