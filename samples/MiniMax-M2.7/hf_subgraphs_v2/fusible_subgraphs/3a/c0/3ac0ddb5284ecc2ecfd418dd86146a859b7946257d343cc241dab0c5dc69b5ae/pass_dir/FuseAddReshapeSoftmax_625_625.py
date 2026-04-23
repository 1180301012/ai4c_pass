import torch
import triton
import triton.language as tl


# Autotune configurations for different sizes (625 is larger, needs more threads)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_add_softmax_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    tmp_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized fused kernel: add + reshape + softmax
    
    This kernel processes each row independently.
    in_0: [1, 1, M, N] - broadcast
    in_1: [1, S, M, N] - main tensor, S=8
    out: [S, M, N]
    tmp: [1, S, M, N]
    """
    pid = tl.program_id(0)
    
    # Each program handles one row: position [b, s, m] -> N elements
    batch = pid // (M * S)
    remainder = pid % (M * S)
    seq = remainder // M
    row = remainder % M
    
    # Compute offsets for loading
    row_start = row * N
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load in_0 [1, 1, M, N], broadcast
    in_0 = tl.load(in_0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Load in_1 [1, S, M, N]: flat_idx = s*M*N + m*N + n
    in_1_offset = seq * M * N + row_start + tl.arange(0, BLOCK_SIZE)
    in_1 = tl.load(in_1_ptr + in_1_offset, mask=mask, other=0.0).to(tl.float32)
    
    # Add with broadcast
    sum_vals = in_0 + in_1
    
    # Compute softmax using online stable algorithm
    max_val = tl.max(sum_vals, axis=0)
    exp_vals = tl.exp(sum_vals - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / (sum_exp + 1e-10)
    
    # Store out [S, M, N]
    out_offset = seq * M * N + row_start + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + out_offset, softmax_vals, mask=mask)
    
    # Store tmp [1, S, M, N]
    tl.store(tmp_ptr + in_1_offset, softmax_vals, mask=mask)


@torch.fx.wrap
def triton_fused_add_softmax(in_0, in_1):
    """Fused add + softmax with proper reshaping"""
    dtype = in_1.dtype
    device = in_1.device
    
    # in_0: [1, 1, M, N], in_1: [1, 8, M, N]
    B0, S, M, N = in_1.shape
    
    # Ensure contiguous memory
    in_0 = in_0.contiguous()
    in_1 = in_1.contiguous()
    
    # Output shapes
    # out: [8, M, N] = [S, M, N]
    # tmp: [1, 8, M, N] = [1, S, M, N]
    
    out = torch.empty([S, M, N], dtype=dtype, device=device)
    tmp = torch.empty([1, S, M, N], dtype=dtype, device=device)
    
    # Total rows to process: S * M
    num_rows = S * M
    grid = (num_rows,)
    
    fused_add_softmax_kernel[grid](
        in_0, in_1, out, tmp, M, N, S
    )
    
    return out, tmp


def pattern(in_0, in_1):
    """
    Match the computation for 8x625x625 shape:
    tmp_0 = in_1 + in_0  # element-wise add with broadcast
    tmp_1 = tmp_0.view(8, 625, 625)  # reshape
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.view(1, 8, 625, 625)
    tmp_4 = tmp_3.view(8, 625, 625)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.0, training=False)
    return tmp_5, tmp_3
    """
    tmp_0 = in_1 + in_0
    tmp_1 = tmp_0.view(8, 625, 625)
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.view(1, 8, 625, 625)
    tmp_4 = tmp_3.view(8, 625, 625)
    tmp_5 = torch.nn.functional.dropout(tmp_4, p=0.0, training=False)
    return tmp_5, tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return triton_fused_add_softmax