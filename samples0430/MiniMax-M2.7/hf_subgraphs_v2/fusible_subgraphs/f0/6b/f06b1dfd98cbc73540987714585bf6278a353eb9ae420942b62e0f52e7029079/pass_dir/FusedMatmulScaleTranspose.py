import torch
import triton
import triton.language as tl


@triton.jit
def fused_matmul_scale_transpose_kernel(
    text_embeds_ptr,  # in_2: shape [2, 1024]
    t_ptr,            # in_1: shape [1024, 1]
    logit_scale,      # in_0: scalar
    out_ptr,          # tmp_1 output: shape [2, 1]
    out_transpose_ptr, # tmp_2 output: shape [1, 2]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for:
    1. matmul: [M, K] @ [K, N] -> [M, N]
    2. scale: * logit_scale
    3. output both [M, N] and [N, M] (transpose)
    
    For this specific case:
    - M = 2, K = 1024, N = 1
    """
    # Get program id
    pid = tl.program_id(0)
    
    # Each program handles one row of output [M, N]
    if pid >= M:
        return
    
    # Compute row and col offsets for this program
    row_offsets = pid * N + tl.arange(0, N)
    
    # Initialize accumulator
    acc = tl.zeros([N], dtype=tl.float32)
    
    # Loop over K dimension with blocking
    for k in range(0, K, BLOCK_SIZE):
        k_offsets = k + tl.arange(0, BLOCK_SIZE)
        
        # Mask for K dimension
        k_mask = k_offsets < K
        
        # Load from text_embeds: [M, K] - get row pid
        a_offsets = pid * K + k_offsets
        a = tl.load(text_embeds_ptr + a_offsets, mask=k_mask, other=0.0)
        
        # Load from t: [K, N] - all columns
        b_offsets = k_offsets * N + tl.arange(0, N)
        b = tl.load(t_ptr + b_offsets, mask=k_mask, other=0.0)
        
        # Multiply and accumulate
        acc += tl.sum(a[:, None] * b[None, :], axis=0)
    
    # Scale by logit_scale
    acc = acc * logit_scale
    
    # Store output [M, N]
    tl.store(out_ptr + row_offsets, acc)
    
    # Store transpose [N, M] - each program handles one column of transpose
    # For N=1 case, store to out_transpose_ptr[0, pid] = acc[0]
    col_offsets = tl.arange(0, M)
    tl.store(out_transpose_ptr + col_offsets, acc[0])


@torch.fx.wrap
def fused_matmul_scale_transpose(text_embeds, t, logit_scale):
    """
    Fused kernel: matmul + scale + transpose
    
    Args:
        text_embeds: [2, 1024] - in_2
        t: [1024, 1] - in_1
        logit_scale: scalar tensor - in_0
    
    Returns:
        tmp_1: [2, 1] - matmul * scale
        tmp_2: [1, 2] - transpose of tmp_1
    """
    M = text_embeds.shape[0]  # 2
    K = text_embeds.shape[1]  # 1024
    N = t.shape[1]            # 1
    
    # Allocate output tensors
    tmp_1 = torch.empty((M, N), dtype=text_embeds.dtype, device=text_embeds.device)
    tmp_2 = torch.empty((N, M), dtype=text_embeds.dtype, device=text_embeds.device)
    
    # Block size for K dimension
    BLOCK_SIZE = 1024
    
    # Launch kernel with M programs
    grid = (M,)
    
    fused_matmul_scale_transpose_kernel[grid](
        text_embeds, t, logit_scale,
        tmp_1, tmp_2,
        M, N, K,
        BLOCK_SIZE
    )
    
    return tmp_1, tmp_2


def pattern(in_0, in_1, in_2):
    """
    Match the pattern: matmul + scale + transpose
    in_0: logit_scale (scalar)
    in_1: t (1024, 1)
    in_2: text_embeds_2 (2, 1024)
    """
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    tmp_2 = tmp_1.t()
    return tmp_1, tmp_2


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_matmul_scale_transpose