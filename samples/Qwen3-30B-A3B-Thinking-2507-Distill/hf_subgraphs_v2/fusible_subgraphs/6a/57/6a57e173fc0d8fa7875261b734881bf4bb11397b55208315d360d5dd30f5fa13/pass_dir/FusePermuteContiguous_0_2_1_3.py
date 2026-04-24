import torch
import triton
import triton.language as tl


def pattern(x):
    tmp = x.permute(0, 2, 1, 3)
    out = tmp.contiguous()
    return out


def replacement_args(x):
    return (x,)


# Grid = (B * H * C,): each program handles one (b, h, c) group's D elements.
# This ensures both the read (input[b,c,h,0..D-1]) and write
# (output[b,h,c,0..D-1]) are perfectly coalesced in memory.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 8},  num_warps=1),
        triton.Config({'BLOCK_D': 16}, num_warps=1),
        triton.Config({'BLOCK_D': 32}, num_warps=1),
        triton.Config({'BLOCK_D': 64}, num_warps=2),
        triton.Config({'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def permute_0213_contiguous_kernel(
    in_ptr, out_ptr,
    B, C, H, D,
    BLOCK_D: tl.constexpr,
):
    # Each program is responsible for one (b, h, c) slice of D elements.
    pid = tl.program_id(0)
    CH = C * H
    b_idx = pid // CH
    h_idx = (pid % CH) // C
    c_idx = pid % C

    d_offsets = tl.arange(0, BLOCK_D)
    mask = d_offsets < D

    # Input layout  [B, C, H, D]  – D elements at [b,c,h,:] are contiguous
    in_base = b_idx * (C * H * D) + c_idx * (H * D) + h_idx * D
    # Output layout [B, H, C, D]  – D elements at [b,h,c,:] are contiguous
    out_base = b_idx * (H * C * D) + h_idx * (C * D) + c_idx * D

    val = tl.load(in_ptr + in_base + d_offsets, mask=mask)
    tl.store(out_ptr + out_base + d_offsets, val, mask=mask)


@torch.fx.wrap
def triton_permute_0213_contiguous(x):
    B, C, H, D = x.shape
    out = torch.empty(B, H, C, D, dtype=x.dtype, device=x.device)
    # One program per (b, h, c) group
    grid = (B * H * C,)
    permute_0213_contiguous_kernel[grid](x, out, B, C, H, D)
    return out


def replacement_func():
    return triton_permute_0213_contiguous