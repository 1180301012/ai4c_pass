import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation in model.py
def pattern(in_0, in_1):
    tmp_0 = in_1 * 0.1767766952966369
    tmp_1 = in_0.unsqueeze(2)
    tmp_2 = tmp_0 + tmp_1
    tmp_3 = tmp_2.softmax(dim=-1)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Triton kernel for fused scale + mask + softmax
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['softmax_dim'],
)
@triton.jit
def fused_scaled_masked_softmax_kernel(
    in_0_ptr,       # Mask: [1, H, N, N]
    in_1_ptr,       # Scores: [1, H, C, N, N]
    out_ptr,        # Output: [1, H, C, N, N]
    scale,          # Scale factor
    H,              # 361
    C,              # 3
    softmax_dim,    # 49 (softmax dimension)
    stride_mask_h,  # H dimension stride for mask
    stride_mask_i,  # i dimension stride for mask
    stride_in_h,    # H dimension stride for scores
    stride_in_c,    # C dimension stride for scores
    stride_in_i,    # i dimension stride for scores
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the softmax
    # Total rows: H * C * softmax_dim
    row_idx = tl.program_id(0)
    
    # Decompose row_idx into (h, c, i) indices
    i = row_idx % softmax_dim
    tmp = row_idx // softmax_dim
    c = tmp % C
    h = tmp // C
    
    # Calculate offsets
    # Mask offset: in_0[0, h, i, :]
    mask_offset = h * stride_mask_h + i * stride_mask_i
    
    # Score offset: in_1[0, h, c, i, :]
    score_offset = h * stride_in_h + c * stride_in_c + i * stride_in_i
    
    # Load data
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < softmax_dim
    
    # Load mask and scores
    mask_vals = tl.load(in_0_ptr + mask_offset + offs, mask=mask, other=0.0)
    score_vals = tl.load(in_1_ptr + score_offset + offs, mask=mask, other=0.0)
    
    # Scale scores and add mask
    x = score_vals * scale + mask_vals
    
    # Numerically stable softmax
    # Step 1: Find max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max
    
    # Step 2: Compute exp
    x_exp = tl.exp(x)
    
    # Step 3: Compute sum of exp (with masking for valid elements)
    x_exp_masked = tl.where(mask, x_exp, 0.0)
    x_sum = tl.sum(x_exp_masked, axis=0)
    
    # Step 4: Normalize
    out = x_exp / x_sum
    
    # Store result
    tl.store(out_ptr + score_offset + offs, out, mask=mask)

# Wrapper function for the kernel
@torch.fx.wrap
def fused_scaled_masked_softmax(in_0, in_1):
    """
    Fused scale + mask + softmax operation.
    
    Args:
        in_0: Mask tensor [1, H, N, N]
        in_1: Score tensor [1, H, C, N, N]
    
    Returns:
        Output tensor [1, H, C, N, N] after softmax
    """
    # Get dimensions
    B, H, C, N, _ = in_1.shape
    
    # Scale factor
    scale = 0.1767766952966369
    
    # Allocate output
    out = torch.empty_like(in_1)
    
    # Total number of rows to process
    num_rows = H * C * N
    
    # Calculate strides
    stride_mask_h = in_0.stride(1)
    stride_mask_i = in_0.stride(2)
    stride_in_h = in_1.stride(1)
    stride_in_c = in_1.stride(2)
    stride_in_i = in_1.stride(3)
    
    # Launch kernel
    fused_scaled_masked_softmax_kernel[(num_rows,)](
        in_0,
        in_1,
        out,
        scale,
        H,
        C,
        N,  # softmax_dim
        stride_mask_h,
        stride_mask_i,
        stride_in_h,
        stride_in_c,
        stride_in_i,
    )
    
    return out

# Replacement function - returns the function reference
def replacement_func():
    return fused_scaled_masked_softmax