import torch
import triton
import triton.language as tl

# Pattern matching function - matches add + permute pattern (excluding view for now)
def pattern(in_0, in_1):
    """
    Define the computation pattern to match:
    1. tmp_0 = in_1 + in_0 (element-wise addition)
    2. tmp_1 = tmp_0.permute(0, 2, 1) (transpose dimensions)
    
    The view operation is handled separately or by the replacement function.
    We need to also return the view result to match the model's return.
    """
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.permute(0, 2, 1)
    # Return the permuted tensor - the view will be handled by our kernel
    return tmp_1

# Argument extraction function
def replacement_args(in_0, in_1):
    # Extract and return arguments needed for the replacement
    return (in_0, in_1)

# Optimized Triton kernel that fuses add + permute + view
@triton.jit
def fused_add_permute_view_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    B: tl.constexpr,      # batch size (1)
    M: tl.constexpr,      # first dimension after permute
    N: tl.constexpr,      # second dimension after permute
    K: tl.constexpr,      # original middle dimension
    stride_in_0_b: tl.constexpr,
    stride_in_0_m: tl.constexpr,
    stride_in_0_k: tl.constexpr,
    stride_in_1_b: tl.constexpr,
    stride_in_1_m: tl.constexpr,
    stride_in_1_k: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_out_n1: tl.constexpr,
    stride_out_n2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Element-wise addition of in_0 and in_1
    2. Permute (transpose from [B, K, M] to [B, M, K])
    3. View (reshape from [B, M, K] to [B, M, sqrt(K), sqrt(K)])
    
    Input shapes: [B, K, M] where K = N (since N*N = K after view)
    Output shape: [B, M, N, N]
    """
    # Calculate total elements in input
    total_elements = B * M * K
    
    # Get program ID and block offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset range for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Calculate 3D indices from flat offset
    # Original tensor is [B, K, M], we need to handle it as [B, M, K] after permute
    # But inputs are in [B, K, M] layout, we permute to [B, M, K]
    b_offsets = offsets // (M * K)
    remainder = offsets % (M * K)
    k_offsets = remainder // M
    m_offsets = remainder % M
    
    # Load from in_0 [B, K, M] at position [b, k, m]
    in_0_idx = b_offsets * stride_in_0_b + k_offsets * stride_in_0_k + m_offsets * stride_in_0_m
    x0 = tl.load(in_0_ptr + in_0_idx, mask=mask, other=0.0)
    
    # Load from in_1 [B, K, M] at position [b, k, m]
    in_1_idx = b_offsets * stride_in_1_b + k_offsets * stride_in_1_k + m_offsets * stride_in_1_m
    x1 = tl.load(in_1_ptr + in_1_idx, mask=mask, other=0.0)
    
    # Perform addition
    result = x0 + x1
    
    # Store to output [B, M, N, N]
    # After permute: [B, K, M] -> [B, M, K]
    # After view: [B, M, K] -> [B, M, N, N] where N*N = K
    # Output indices: [b, m, n1, n2]
    # We need to map flat offset to [b, m, n1, n2]
    # Since offsets are for original [B, K, M], and after permute we have [B, M, K]
    # We need to compute n1, n2 from k and m
    
    # k_offsets is the K dimension, we need to convert to (n1, n2)
    # k = n1 * N + n2
    N_val = N  # Sqrt of K
    n1_offsets = k_offsets // N_val
    n2_offsets = k_offsets % N_val
    
    out_idx = (b_offsets * stride_out_b + 
               m_offsets * stride_out_m + 
               n1_offsets * stride_out_n1 + 
               n2_offsets * stride_out_n2)
    
    tl.store(out_ptr + out_idx, result, mask=mask)


@torch.fx.wrap
def fused_add_permute_view_wrapper(in_0, in_1):
    """
    Wrapper function that launches the fused kernel.
    
    Input shapes: [B, K, M] each (e.g., [1, 9216, 64] or [1, 2304, 192])
    Output shape: [B, M, N, N] where N*N = K (e.g., [1, 64, 96, 96] or [1, 192, 48, 48])
    """
    B, K, M = in_0.shape  # in_0 is [B, K, M]
    N = int(K ** 0.5)     # K should be a perfect square (9216=96*96, 2304=48*48)
    
    # Create output tensor with correct shape
    out = torch.empty((B, M, N, N), dtype=in_0.dtype, device=in_0.device)
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate total elements
    total_elements = B * M * K
    
    # Calculate grid size
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_add_permute_view_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        B=B,
        M=M,
        N=N,
        K=K,
        stride_in_0_b=in_0.stride(0),
        stride_in_0_m=in_0.stride(2),
        stride_in_0_k=in_0.stride(1),
        stride_in_1_b=in_1.stride(0),
        stride_in_1_m=in_1.stride(2),
        stride_in_1_k=in_1.stride(1),
        stride_out_b=out.stride(0),
        stride_out_m=out.stride(1),
        stride_out_n1=out.stride(2),
        stride_out_n2=out.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_add_permute_view_wrapper