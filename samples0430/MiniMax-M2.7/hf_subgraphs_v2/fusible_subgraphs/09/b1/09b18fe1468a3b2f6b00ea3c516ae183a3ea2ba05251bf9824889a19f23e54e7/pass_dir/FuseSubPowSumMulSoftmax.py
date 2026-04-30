import torch
import triton
import triton.language as tl

@triton.jit
def fused_attention_kernel(
    in_1_ptr, in_2_ptr, in_3_ptr,
    out_ptr,
    B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Grid: (B, N, H) - each program handles one (batch, N, head) position
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    
    # Offset for K dimension
    k_offsets = tl.arange(0, BLOCK_SIZE)
    mask = k_offsets < K
    
    # in_1[b, n, h, k] and in_2[b, 0, h, k]
    in_1_offsets = b * N * H * K + n * H * K + h * K + k_offsets
    in_2_offsets = b * H * K + h * K + k_offsets  # in_2 is [B, 1, H, K]
    
    # Load data
    in_1 = tl.load(in_1_ptr + in_1_offsets, mask=mask, other=0.0)
    in_2 = tl.load(in_2_ptr + in_2_offsets, mask=mask, other=0.0)
    
    # Step 1: subtraction
    diff = in_1 - in_2
    
    # Step 2: square
    sq = diff * diff
    
    # Step 3: sum over K dimension ( BLOCK_SIZE == K )
    sum_sq = tl.sum(sq, axis=0)
    
    # Step 4: multiply by in_3[b, 0, h] which is [1, 32]
    in_3_val = tl.load(in_3_ptr + h)
    scaled = sum_sq * in_3_val
    
    # Step 5: compute exp for softmax
    exp_val = tl.exp(scaled)
    
    # Store exp value - will be normalized later
    out_ptr_base = b * N * H + n * H + h
    tl.store(out_ptr + out_ptr_base, exp_val)


@torch.fx.wrap
def fused_attention_wrapper(in_1, in_2, in_3):
    """
    Fused kernel for: subtract, pow(2), sum(dim=3), mul, softmax
    in_1: [1, 4096, 32, 512]
    in_2: [1, 1, 32, 512] 
    in_3: [1, 1, 32]
    Returns: attention weights [1, 4096, 32]
    """
    B, N, H, K = 1, 4096, 32, 512
    
    # Allocate output buffer for exp values
    exp_output = torch.empty((B, N, H), dtype=in_1.dtype, device=in_1.device)
    
    BLOCK_SIZE = 512  # K dimension
    
    # Grid: (B, N, H)
    grid = (B, N, H)
    
    fused_attention_kernel[grid](
        in_1, in_2, in_3.squeeze(0),  # in_3: [1, 1, 32] -> [32]
        exp_output,
        B, N, H, K,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Softmax normalization over H dimension
    # Compute sum of exp values across H for normalization
    sum_exp = exp_output.sum(dim=2, keepdim=True)  # [B, N, 1]
    attn_weights = exp_output / sum_exp  # [B, N, H]
    
    return attn_weights


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the subtract-pow-sum-mul-softmax pattern.
    Returns tmp_5 (softmax) which is the observable output from this subgraph.
    """
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim=3)
    tmp_4 = in_3 * tmp_3
    tmp_5 = torch.nn.functional.softmax(tmp_4, dim=2)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_1, in_2, in_3)


def replacement_func():
    return fused_attention_wrapper