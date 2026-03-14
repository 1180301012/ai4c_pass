import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Full pattern for 24x24: add + split + getitem + permute + view
    This matches the complete computation pattern.
    """
    tmp_0 = in_1 + in_0
    tmp_1 = torch.functional.split(tmp_0, [1, 576], 1)
    tmp_2 = tmp_1[0]
    tmp_3 = tmp_1[1]
    tmp_4 = tmp_3.permute(0, 2, 1)
    tmp_5 = tmp_4.view(1, 384, 24, 24)
    return tmp_2, tmp_5


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
def fused_add_split_permute_kernel_24(
    in0_ptr, in1_ptr,
    out1_ptr,  # class token: [1, 1, 384]
    out2_ptr,  # patches after permute: [1, 384, 24, 24]
    M, N,  # M=577 (seq_len), N=384 (hidden_dim)
    stride_in_m, stride_in_n,
    stride_out1_m, stride_out1_n,
    stride_out2_c, stride_out2_h, stride_out2_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Fused kernel: add + split + permute + view
    - Reads in0, in1 once
    - Writes to out1 (class token) and out2 (permuted patches)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Create mask
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # Compute pointers
    in0_ptrs = in0_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n
    in1_ptrs = in1_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n
    
    # Load and add
    in0 = tl.load(in0_ptrs, mask=mask, other=0.0)
    in1 = tl.load(in1_ptrs, mask=mask, other=0.0)
    added = in0 + in1
    
    # Split: first token goes to out1, rest goes to out2 (after permute)
    # For seq_idx == 0 (first token): write to out1
    is_first_token = offs_m[:, None] == 0
    out1_ptrs = out1_ptr + offs_n[None, :]
    tl.store(out1_ptrs, added, mask=mask & is_first_token)
    
    # For seq_idx > 0 (patches): permute and write to out2
    # Input shape for patches: [576, 384]
    # After permute: [384, 576] which we view as [384, 24, 24]
    is_patch = offs_m[:, None] > 0
    patch_idx = offs_m[:, None] - 1  # 0..575
    hidden_idx = offs_n[None, :]     # 0..383
    
    # After permute(0, 2, 1): hidden becomes dim 0, patch_idx becomes dim 1
    # Linear offset: hidden_idx * 576 + patch_idx
    out2_offset = hidden_idx * 576 + patch_idx
    out2_ptrs = out2_ptr + out2_offset
    tl.store(out2_ptrs, added, mask=mask & is_patch)


@torch.fx.wrap
def fused_add_split_permute_view_24(in_0, in_1):
    """Fused implementation for 577x384 -> (1x384, 384x24x24)"""
    batch, seq_len, hidden = in_0.shape  # [1, 577, 384]
    
    # Allocate outputs
    out1 = torch.empty((batch, 1, hidden), device=in_0.device, dtype=in_0.dtype)
    out2 = torch.empty((batch, hidden, 24, 24), device=in_0.device, dtype=in_0.dtype)
    
    # Launch kernel with 2D grid
    grid = lambda META: (
        triton.cdiv(seq_len, META['BLOCK_SIZE_M']),
        triton.cdiv(hidden, META['BLOCK_SIZE_N']),
    )
    
    fused_add_split_permute_kernel_24[grid](
        in_0, in_1,
        out1, out2,
        seq_len, hidden,
        in_0.stride(1), in_0.stride(2),
        out1.stride(1), out1.stride(2),
        out2.stride(1), out2.stride(2), out2.stride(3),
    )
    
    return out1, out2


def replacement_func():
    return fused_add_split_permute_view_24