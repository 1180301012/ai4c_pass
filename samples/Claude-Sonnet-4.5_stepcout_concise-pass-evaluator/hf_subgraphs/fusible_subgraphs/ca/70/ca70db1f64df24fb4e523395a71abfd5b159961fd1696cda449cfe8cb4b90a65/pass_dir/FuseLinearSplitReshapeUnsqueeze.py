import torch
import triton
import triton.language as tl

def pattern(in_5, in_1, in_0):
    """
    Pattern: linear -> split into two halves -> reshape -> unsqueeze (for first half)
    Outputs: tmp_8 (second half), tmp_13 (first half unsqueezed)
    """
    tmp_4 = torch.nn.functional.linear(in_5, in_1, in_0)
    tmp_5 = tmp_4[slice(None, None, None), slice(None, 256, None)]
    tmp_6 = tmp_5.view(-1, 256)
    tmp_7 = tmp_4[slice(None, None, None), slice(-256, None, None)]
    tmp_8 = tmp_7.view(-1, 256)
    tmp_13 = tmp_6.unsqueeze(-2)
    return tmp_8, tmp_13

def replacement_args(in_5, in_1, in_0):
    return (in_5, in_1, in_0)

@triton.jit
def fused_linear_split_kernel(
    input_ptr, weight_ptr, bias_ptr,
    out_first_ptr, out_second_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused linear + split kernel
    input: [M, K]
    weight: [N, K] (will be transposed)
    bias: [N]
    out_first: [M, 1, N//2] (first half with unsqueeze)
    out_second: [M, N//2] (second half)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute output position
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Determine which half we're computing
    half_size = N // 2
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Matrix multiplication
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        # Load input
        input_mask = (rm[:, None] < M) & (rk[None, :] < K)
        input_block = tl.load(input_ptr + rm[:, None] * K + rk[None, :], mask=input_mask, other=0.0)
        
        # Load weight (transposed)
        weight_mask = (rn[:, None] < N) & (rk[None, :] < K)
        weight_block = tl.load(weight_ptr + rn[:, None] * K + rk[None, :], mask=weight_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(input_block, tl.trans(weight_block))
    
    # Add bias
    bias_mask = rn < N
    bias_block = tl.load(bias_ptr + rn, mask=bias_mask, other=0.0)
    acc += bias_block[None, :]
    
    # Store to appropriate output
    output_mask = (rm[:, None] < M) & (rn[None, :] < N)
    
    if pid_n < tl.cdiv(half_size, BLOCK_N):
        # First half - store with unsqueeze dimension
        out_rn = rn
        out_mask = (rm[:, None] < M) & (out_rn[None, :] < half_size)
        # Store as [M, 1, N//2] by computing offset: m * (1 * half_size) + 0 * half_size + n
        tl.store(out_first_ptr + rm[:, None] * half_size + out_rn[None, :], acc, mask=out_mask)
    else:
        # Second half
        out_rn = rn - half_size
        out_mask = (rm[:, None] < M) & (out_rn[None, :] < half_size) & (out_rn[None, :] >= 0)
        tl.store(out_second_ptr + rm[:, None] * half_size + out_rn[None, :], acc, mask=out_mask)

@torch.fx.wrap
def fused_linear_split_unsqueeze(input, weight, bias):
    M, K = input.shape
    N, K_w = weight.shape
    assert K == K_w, f"Dimension mismatch: {K} vs {K_w}"
    
    half_size = N // 2
    
    # Output shapes
    out_first = torch.empty((M, 1, half_size), device=input.device, dtype=input.dtype)
    out_second = torch.empty((M, half_size), device=input.device, dtype=input.dtype)
    
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 32
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    fused_linear_split_kernel[grid](
        input, weight, bias,
        out_first, out_second,
        M, N, K,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    return out_second, out_first

def replacement_func():
    return fused_linear_split_unsqueeze