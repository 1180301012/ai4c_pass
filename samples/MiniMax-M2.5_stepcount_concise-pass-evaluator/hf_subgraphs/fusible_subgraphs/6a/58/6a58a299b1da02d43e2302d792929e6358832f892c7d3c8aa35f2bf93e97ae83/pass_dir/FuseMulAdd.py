import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
    ],
    key=['N_total'],
)
@triton.jit
def triton_fused_mul_add_kernel(
    in_2_ptr, in_1_ptr, in_0_ptr, out_ptr,
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_in_2_b, stride_in_2_m,
    N_total: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: out = in_2 * in_1 + in_0
    
    Uses 1D grid where each block processes BLOCK_SIZE elements.
    Total elements = B * M * N * K = B * M * 2 * K
    
    Input shapes:
    - in_2: [B, M, 1, K] -> squeezed to [B, M, K]
    - in_1: [1, 1, 2, K] -> squeezed to [2, K]
    - in_0: [2, K]
    Output: [B, M, 2, K]
    """
    # 1D grid - each thread processes one element
    pid = tl.program_id(0)
    
    # Calculate offset
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_total
    
    # Calculate b, m, n, k indices
    # N_total = B * M * N * K = B * M * 2 * K
    # For each element, we compute: out[b, m, n, k]
    
    # Use modular arithmetic to calculate indices
    # This is equivalent to:
    # b = offsets // (M * N * K)
    # remainder = offsets % (M * N * K)
    # m = remainder // (N * K)
    # remainder2 = remainder % (N * K)
    # n = remainder2 // K
    # k = remainder2 % K
    
    b = offsets // (M * N * K)
    rem1 = offsets % (M * N * K)
    m = rem1 // (N * K)
    rem2 = rem1 % (N * K)
    n = rem2 // K
    k = rem2 % K
    
    # Bounds check for b
    valid_b = b < B
    mask = mask & valid_b
    
    # Calculate offsets for loading from in_2 [B, M, K]
    # in_2[b, m, k] = b * stride_in_2_b + m * stride_in_2_m + k
    in_2_offset = b * stride_in_2_b + m * stride_in_2_m + k
    val_in_2 = tl.load(in_2_ptr + in_2_offset, mask=mask, other=0.0)
    
    # Load in_1[n, k]
    in_1_offset = n * K + k
    val_in_1 = tl.load(in_1_ptr + in_1_offset, mask=mask, other=0.0)
    
    # Load in_0[n, k]
    val_in_0 = tl.load(in_0_ptr + in_1_offset, mask=mask, other=0.0)
    
    # Compute fused multiply-add
    out_val = val_in_2 * val_in_1 + val_in_0
    
    # Store to output [B, M, N, K]
    out_offset = b * M * N * K + m * N * K + n * K + k
    tl.store(out_ptr + out_offset, out_val, mask=mask)


@torch.fx.wrap
def fused_mul_add(in_0, in_1, in_2):
    """
    Fused multiply-add operation: out = in_2 * in_1 + in_0
    
    Handles broadcasting for in_2 [B, M, 1, K], in_1 [1, 1, 2, K], in_0 [2, K]
    Result should be [B, M, 2, K]
    """
    # Get shapes
    B = in_2.shape[0]
    M = in_2.shape[1]
    N = 2  # The unbind dimension
    K = in_2.shape[3]
    
    # Squeeze in_2 to [B, M, K]
    in_2_squeezed = in_2.squeeze(2)
    
    # Squeeze in_1 from [1, 1, 2, K] to [2, K]
    in_1_squeezed = in_1.squeeze(0).squeeze(0)
    
    # Output shape
    out = torch.empty((B, M, N, K), dtype=in_0.dtype, device=in_0.device)
    
    # Total number of elements to process
    N_total = B * M * N * K
    
    # Calculate 1D grid
    # Use num_warps=4 for better occupancy
    grid = ((N_total + 255) // 256,)
    
    triton_fused_mul_add_kernel[grid](
        in_2_squeezed, in_1_squeezed, in_0, out,
        B, M, N, K,
        in_2_squeezed.stride(0), in_2_squeezed.stride(1),
        N_total,
    )
    
    return out


def pattern(in_0, in_1, in_2):
    """
    Match: tmp_1 = in_2 * in_1; tmp_2 = tmp_1 + in_0
    This fuses the multiply and add operations.
    """
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    return tmp_2


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_mul_add