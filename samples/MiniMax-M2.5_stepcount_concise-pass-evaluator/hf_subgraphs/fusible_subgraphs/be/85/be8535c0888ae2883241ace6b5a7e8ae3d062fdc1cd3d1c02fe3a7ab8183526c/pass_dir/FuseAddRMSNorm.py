import torch
import triton
import triton.language as tl

# Pattern matching function - matches the computation to optimize
# This includes: add + to(float32) + pow(2) + mean + add(eps) + rsqrt + multiply + multiply
def pattern(in_0, in_1, in_2, in_3):
    # Match the full computation pattern
    tmp_1 = in_3 + in_2
    tmp_4 = tmp_1.to(torch.float32)
    tmp_5 = tmp_4.pow(2)
    tmp_6 = tmp_5.mean(-1, keepdim=True)
    tmp_7 = tmp_6 + 1e-06
    tmp_8 = torch.rsqrt(tmp_7)
    tmp_9 = tmp_1 * tmp_8
    tmp_10 = in_0 * tmp_9
    return tmp_1, tmp_10


# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    # Return all arguments needed for the replacement
    return (in_0, in_1, in_2, in_3)


# Autotune configurations for better performance
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_rmsnorm_kernel(
    in_0_ptr,      # weight tensor [feature_dim]
    in_1_ptr,      # arange (not used in computation but needed for signature)
    in_2_ptr,      # dropout tensor [batch, seq, feature]
    in_3_ptr,      # hidden_states tensor [batch, seq, feature]
    out_1_ptr,     # output tmp_1 (the addition result)
    out_2_ptr,     # output tmp_10 (final result)
    N,             # feature dimension
    M,             # batch * seq
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Fused kernel that computes RMSNorm with fused addition
    # Each program processes one element in the sequence dimension (M)
    # We need to compute:
    # 1. tmp_1 = in_3 + in_2 (per element)
    # 2. tmp_10 = in_0 * tmp_1 * rsqrt(mean(tmp_1^2) + eps)
    
    row_idx = tl.program_id(0)
    
    # Compute the offset for this row
    row_offset = row_idx * N
    
    # Load in_0 (weight) - broadcast across the row
    weight = tl.load(in_0_ptr + tl.arange(0, BLOCK_SIZE))
    weight = tl.where(tl.arange(0, BLOCK_SIZE) < N, weight, 0.0)
    
    # Load in_2 (dropout) and in_3 (hidden_states) for this row
    # and compute tmp_1 = in_3 + in_2
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_offset + N
    
    dropout = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    hidden = tl.load(in_3_ptr + offsets, mask=mask, other=0.0)
    
    # tmp_1 = in_3 + in_2
    tmp_1 = dropout + hidden
    
    # Store tmp_1 to output
    tl.store(out_1_ptr + offsets, tmp_1, mask=mask)
    
    # Compute tmp_1^2 for reduction
    tmp_1_sq = tmp_1 * tmp_1
    
    # Compute sum of squares using reduction
    # Sum along the feature dimension
    sum_of_squares = tl.sum(tl.where(offsets < row_offset + N, tmp_1_sq, 0.0), axis=0)
    
    # Compute mean = sum / N
    mean = sum_of_squares / N
    
    # Compute rsqrt(mean + eps)
    normalization_factor = tl.rsqrt(mean + eps)
    
    # Compute tmp_10 = in_0 * tmp_1 * normalization_factor
    # We need to broadcast weight across the row
    tmp_10 = tmp_1 * weight * normalization_factor
    
    # Store final result
    tl.store(out_2_ptr + offsets, tmp_10, mask=mask)


@torch.fx.wrap
def fused_rmsnorm_wrapper(in_0, in_1, in_2, in_3):
    # Wrapper function that launches the fused Triton kernel
    # Get shapes
    batch, seq, feature = in_2.shape  # or in_3.shape (they're the same)
    
    # Allocate output tensors
    tmp_1 = torch.empty_like(in_2)  # Same shape as in_2/in_3
    tmp_10 = torch.empty_like(in_2)  # Same shape as in_2/in_3
    
    # Determine block size based on feature dimension
    # Use power of 2 for efficient memory access
    N = feature
    M = batch * seq
    
    # Choose block size
    BLOCK_SIZE = 1024
    if N <= 64:
        BLOCK_SIZE = 128
    elif N <= 256:
        BLOCK_SIZE = 256
    elif N <= 512:
        BLOCK_SIZE = 512
    elif N <= 1024:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    # Grid: one program per row (M programs)
    grid = (M,)
    
    eps = 1e-06
    
    # Launch kernel
    fused_rmsnorm_kernel[grid](
        in_0,           # weight
        in_1,           # arange (not used)
        in_2,           # dropout
        in_3,           # hidden_states
        tmp_1,          # output 1
        tmp_10,         # output 2
        N,
        M,
        eps,
        BLOCK_SIZE,
    )
    
    return tmp_1, tmp_10


def replacement_func():
    return fused_rmsnorm_wrapper