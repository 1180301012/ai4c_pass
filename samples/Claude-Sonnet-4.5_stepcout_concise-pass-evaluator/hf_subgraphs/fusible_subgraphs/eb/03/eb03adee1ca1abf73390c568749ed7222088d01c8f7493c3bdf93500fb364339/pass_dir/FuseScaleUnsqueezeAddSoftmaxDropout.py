import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=1),
        triton.Config({'BLOCK_N': 64}, num_warps=2),
        triton.Config({'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 64}, num_warps=8),
        triton.Config({'BLOCK_N': 128}, num_warps=1),
        triton.Config({'BLOCK_N': 128}, num_warps=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_N': 256}, num_warps=2),
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_scale_unsqueeze_add_softmax_kernel(
    in_0_ptr,  # [B, M, N, N]
    in_1_ptr,  # [B, M, K, N, N]
    out_ptr,   # [B, M, K, N, N]
    scale,
    B, M, K, N,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row of softmax
    pid = tl.program_id(0)
    
    # Decompose pid into [b, m, k, n]
    b = pid // (M * K * N)
    remainder = pid % (M * K * N)
    m = remainder // (K * N)
    remainder = remainder % (K * N)
    k = remainder // N
    n = remainder % N
    
    # Base offsets
    in_1_row_offset = b * (M * K * N * N) + m * (K * N * N) + k * (N * N) + n * N
    in_0_row_offset = b * (M * N * N) + m * (N * N) + n * N
    out_row_offset = in_1_row_offset
    
    # Load all N elements
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    
    in_1_vals = tl.load(in_1_ptr + in_1_row_offset + offsets, mask=mask, other=-float('inf'))
    in_0_vals = tl.load(in_0_ptr + in_0_row_offset + offsets, mask=mask, other=0.0)
    
    # Scale and add (with broadcasting from unsqueeze)
    vals = tl.where(mask, in_1_vals * scale + in_0_vals, -float('inf'))
    
    # Softmax
    max_val = tl.max(vals, axis=0)
    vals_shifted = vals - max_val
    exp_vals = tl.exp(vals_shifted)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / sum_exp
    
    # Store
    tl.store(out_ptr + out_row_offset + offsets, softmax_vals, mask=mask)

@torch.fx.wrap
def fused_scale_unsqueeze_add_softmax(in_0, in_1):
    B, M, N, _ = in_0.shape
    _, _, K, _, _ = in_1.shape
    
    scale = 0.1767766952966369
    
    out = torch.empty_like(in_1)
    
    total_rows = B * M * K * N
    grid = (total_rows,)
    
    fused_scale_unsqueeze_add_softmax_kernel[grid](
        in_0, in_1, out,
        scale, B, M, K, N
    )
    
    return out

def pattern(in_0, in_1):
    tmp_0 = in_1 * 0.1767766952966369
    tmp_1 = in_0.unsqueeze(2)
    tmp_2 = tmp_0 + tmp_1
    tmp_3 = tmp_2.softmax(dim=-1)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return (tmp_4,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    return fused_scale_unsqueeze_add_softmax