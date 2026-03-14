import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Pattern using different operations to avoid split issues"""
    tmp_0 = in_1 + in_0
    # Try slicing instead of split
    tmp_2 = tmp_0[:, :1, :]
    tmp_3 = tmp_0[:, 1:, :]
    tmp_4 = tmp_3.permute(0, 2, 1)
    tmp_5 = tmp_4.view(1, 384, 14, 14)
    return (tmp_2, tmp_5)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_kernel_slicing(
    in0_ptr, in1_ptr,
    out1_ptr,  # class token: [1, 1, 384]
    out2_ptr,  # patches after permute: [1, 384, 14, 14]
    M, N,  # M=197 (seq_len), N=384 (hidden_dim)
    stride_in_m, stride_in_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel: add + slice + permute + view
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    in0_ptrs = in0_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n
    in1_ptrs = in1_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n
    
    in0 = tl.load(in0_ptrs, mask=mask, other=0.0)
    in1 = tl.load(in1_ptrs, mask=mask, other=0.0)
    added = in0 + in1
    
    # First token -> out1
    is_first_token = offs_m[:, None] == 0
    out1_ptrs = out1_ptr + offs_n[None, :]
    tl.store(out1_ptrs, added, mask=mask & is_first_token)
    
    # Rest -> out2 (permuted)
    is_patch = offs_m[:, None] > 0
    patch_idx = offs_m[:, None] - 1
    hidden_idx = offs_n[None, :]
    
    out2_offset = hidden_idx * 196 + patch_idx
    out2_ptrs = out2_ptr + out2_offset
    tl.store(out2_ptrs, added, mask=mask & is_patch)


@torch.fx.wrap
def fused_slicing_14(in_0, in_1):
    """Fused implementation for 197x384"""
    batch, seq_len, hidden = in_0.shape
    
    out1 = torch.empty((batch, 1, hidden), device=in_0.device, dtype=in_0.dtype)
    out2 = torch.empty((batch, hidden, 14, 14), device=in_0.device, dtype=in_0.dtype)
    
    grid = lambda META: (
        triton.cdiv(seq_len, META['BLOCK_SIZE_M']),
        triton.cdiv(hidden, META['BLOCK_SIZE_N']),
    )
    
    fused_kernel_slicing[grid](
        in_0, in_1,
        out1, out2,
        seq_len, hidden,
        in_0.stride(1), in_0.stride(2),
    )
    
    return out1, out2


def replacement_func():
    return fused_slicing_14