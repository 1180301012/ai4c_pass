import torch
import triton
import triton.language as tl


# Autotune configurations to find the best block size
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=4),
    ],
    key=['M'],
)
@triton.jit
def fused_attention_elementwise_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr,
    output_ptr,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: ((in_0 + in_3 + in_2) / 8.0) + in_1
    
    Input shapes:
    - in_0: [B, H, N, M] attention_scores
    - in_1: [B, 1, 1, M] extended_attention_mask_2  
    - in_2: [B, H, N, M] relative_position_scores_key
    - in_3: [B, H, N, M] relative_position_scores_query
    
    Output: [B, H, N, M]
    """
    # Grid: (B, H, N)
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    # Calculate offset for this (batch, head, seq) combination
    # Each thread handles a row of size M (the softmax dimension)
    offset = (batch_idx * H * N * M + head_idx * N * M + seq_idx * M)
    
    # Block size should cover the M dimension
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < M
    
    # Load all inputs - each is [M] sized for this specific (batch, head, seq)
    # in_0: [B, H, N, M] -> offset
    in_0 = tl.load(in_0_ptr + offset + offs, mask=mask, other=0.0)
    # in_3: [B, H, N, M] -> same offset layout
    in_3 = tl.load(in_3_ptr + offset + offs, mask=mask, other=0.0)
    # in_2: [B, H, N, M] -> same offset layout  
    in_2 = tl.load(in_2_ptr + offset + offs, mask=mask, other=0.0)
    # in_1: [B, 1, 1, M] -> broadcasted, offset is just seq_idx * M (batch*1*1*M + 0*1*M + 0*M + offs)
    in_1_offset = batch_idx * M + offs  # Shape [B, 1, 1, M] broadcasts
    in_1 = tl.load(in_1_ptr + in_1_offset, mask=mask, other=0.0)
    
    # Fused computation: ((in_0 + in_3 + in_2) / 8.0) + in_1
    # Using FMA for better precision and performance
    tmp = in_0 + in_3 + in_2
    tmp = tmp / 8.0
    out = tmp + in_1
    
    # Store result
    tl.store(output_ptr + offset + offs, out, mask=mask)


@torch.fx.wrap
def fused_attention_elementwise_wrapper(in_0, in_1, in_2, in_3):
    """
    Fused attention score computation: ((in_0 + in_3 + in_2) / 8.0) + in_1
    """
    B, H, N, M = in_0.shape
    
    # Calculate total elements processed
    total_elements = B * H * N * M
    
    # For small workloads, use PyTorch directly to avoid kernel launch overhead
    # Heuristic: if total elements or grid size is too small, PyTorch is faster
    if total_elements <= 32768 or (B * H * N) <= 32:
        # Fall back to PyTorch for small workloads
        tmp = in_0 + in_3 + in_2
        tmp = tmp / 8.0
        return tmp + in_1
    
    # Allocate output
    output = torch.empty_like(in_0)
    
    # Grid: (B, H, N) 
    grid = (B, H, N)
    
    # Use autotuning to find the best block size
    fused_attention_elementwise_kernel[grid](
        in_0, in_1, in_2, in_3,
        output,
        B, H, N, M,
    )
    
    return output


# Pattern matching function
def pattern(in_0, in_1, in_2, in_3):
    """
    Match the element-wise computation pattern:
    tmp_0 = in_0 + in_3
    tmp_1 = tmp_0 + in_2
    tmp_2 = tmp_1 / 8.0
    tmp_3 = tmp_2 + in_1
    """
    tmp_0 = in_0 + in_3
    tmp_1 = tmp_0 + in_2
    tmp_2 = tmp_1 / 8.0
    tmp_3 = tmp_2 + in_1
    return tmp_3


# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_attention_elementwise_wrapper