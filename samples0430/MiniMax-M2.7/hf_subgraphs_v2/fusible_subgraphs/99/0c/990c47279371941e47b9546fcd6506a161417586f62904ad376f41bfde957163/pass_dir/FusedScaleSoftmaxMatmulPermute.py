import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 32}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_softmax_matmul_kernel(
    # Input tensors
    in_0_ptr, in_1_ptr,
    # Output tensor
    out_ptr,
    # Dimensions
    B: tl.constexpr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    # Strides for inputs
    in_0_stride_batch: tl.constexpr, in_0_stride_m: tl.constexpr, in_0_stride_k: tl.constexpr,
    in_1_stride_batch: tl.constexpr, in_1_stride_k: tl.constexpr, in_1_stride_n: tl.constexpr,
    # Strides for output
    out_stride_batch: tl.constexpr, out_stride_n: tl.constexpr, out_stride_m: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr
):
    """
    Fused kernel: scale -> softmax -> matmul -> permute
    Input shapes: in_0 [B, M, K], in_1 [B, K, N]
    Output shape: [B, N, M] (permuted from [B, M, N])
    
    Computes: permute(softmax(SCALE * in_0, dim=-1) @ in_1, (0, 2, 1))
    K is typically small (e.g., 19), so we handle it with explicit loop
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Masks
    m_mask = rm < M
    n_mask = rn < N
    
    # Calculate strides for offset calculation
    in_0_offset = pid_b * in_0_stride_batch
    in_1_offset = pid_b * in_1_stride_batch
    out_offset = pid_b * out_stride_batch
    
    # ==== STEP 1: Compute max for numerical stability ====
    # Find max across K dimension for each row
    max_vals = tl.zeros((BLOCK_M,), dtype=tl.float32) + float('-inf')
    
    for kk in range(K):
        # Load in_0 [M, 1]
        a_ptrs = (in_0_ptr + 
                  in_0_offset + 
                  rm * in_0_stride_m + 
                  kk * in_0_stride_k)
        a = tl.load(a_ptrs, mask=m_mask, other=0.0)
        a_scaled = tl.cast(a, tl.float32) * SCALE
        max_vals = tl.maximum(max_vals, a_scaled)
    
    # ==== STEP 2: Compute softmax (exp) and matmul ====
    # Accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    exp_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for kk in range(K):
        # Load in_0 [M, 1] - scale and compute exp
        a_ptrs = (in_0_ptr + 
                  in_0_offset + 
                  rm * in_0_stride_m + 
                  kk * in_0_stride_k)
        a = tl.load(a_ptrs, mask=m_mask, other=0.0)
        a_scaled = tl.cast(a, tl.float32) * SCALE
        
        # Compute softmax numerator: exp(x - max)
        exp_val = tl.exp(a_scaled - max_vals)
        exp_sum = exp_sum + exp_val
        
        # Load in_1 [1, N] - row kk
        b_ptrs = (in_1_ptr + 
                  in_1_offset + 
                  kk * in_1_stride_k + 
                  rn * in_1_stride_n)
        b = tl.load(b_ptrs, mask=n_mask, other=0.0)
        b = tl.cast(b, tl.float32)
        
        # Accumulate: acc[m, n] += exp_val[m] * b[n]
        acc = acc + exp_val[:, None] * b[None, :]
    
    # Normalize by exp sum
    softmax_out = acc / (exp_sum[:, None] + 1e-10)
    
    # ==== STEP 3: Store permuted result ====
    # Output is [B, N, M] - we need to store with permuted indices
    # softmax_out is [M, N], output is [N, M]
    # out[rn, rm] = softmax_out[rm, rn]
    out_ptrs = (out_ptr + 
                out_offset + 
                rn[:, None] * out_stride_n + 
                rm[None, :] * out_stride_m)
    tl.store(out_ptrs, softmax_out, mask=n_mask[:, None] & m_mask[None, :])


@torch.fx.wrap
def fused_softmax_matmul_wrapper(in_0, in_1):
    """
    Wrapper function for the fused softmax + matmul + permute kernel.
    in_0: [B, M, K] - scaled input (sim_map)
    in_1: [B, K, N] - weight matrix (value_1)
    Returns: [B, N, M] - permuted output
    
    Original computation:
    - tmp_0 = 0.0625 * in_0
    - tmp_1 = softmax(tmp_0, dim=-1)  -> [B, M, K]
    - matmul = tmp_1 @ in_1 -> [B, M, N]
    - out = matmul.permute(0, 2, 1) -> [B, N, M]
    """
    B, M, K = in_0.shape
    _, _, N = in_1.shape
    
    # Allocate output [B, N, M]
    out = torch.empty((B, N, M), dtype=in_0.dtype, device=in_0.device)
    
    # Grid: (B, M_blocks, N_blocks)
    BLOCK_M, BLOCK_N = 64, 64
    num_m_blocks = (M + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (N + BLOCK_N - 1) // BLOCK_N
    grid = (B, num_m_blocks, num_n_blocks)
    
    fused_softmax_matmul_kernel[grid](
        in_0, in_1, out,
        B, M, K, N,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_M, BLOCK_N,
        0.0625  # SCALE = 1/16
    )
    
    return out


def pattern(in_0, in_1):
    """
    Pattern: Scale + Softmax + Matmul + Permute
    This matches the OCRNet attention computation.
    """
    tmp_0 = 0.0625 * in_0
    tmp_1 = torch.nn.functional.softmax(tmp_0, dim=-1)
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3 = matmul.permute(0, 2, 1)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_softmax_matmul_wrapper