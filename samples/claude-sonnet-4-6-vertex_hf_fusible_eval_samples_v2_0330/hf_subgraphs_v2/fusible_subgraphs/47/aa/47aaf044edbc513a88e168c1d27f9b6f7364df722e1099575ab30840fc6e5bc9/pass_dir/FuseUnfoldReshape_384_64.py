"""
Fused optimization pass for the ConvBert sliding-window reshape pattern.
Handles the case with C=384 channels and group_size=64.

Matches: transpose(1,2) -> reshape(1,-1,384,9) -> reshape(-1,64,9)
Input tmp_2 shape [B, 3456, L], Output shape [B*L*6, 64, 9].

The key saving: transpose makes the tensor non-contiguous, so the first
reshape triggers an expensive contiguous copy. We fuse all three into one
Triton kernel that reads from the [B, CK, L] layout directly.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['N_total'],
)
@triton.jit
def transpose_reshape_384_64_kernel(
    in_ptr,    # [B, CK, L]   CK = num_groups * GROUP_SIZE * KERNEL_SIZE
    out_ptr,   # [N_out, GROUP_SIZE, KERNEL_SIZE]   N_out = B * L * num_groups
    B,
    L,
    num_groups,
    stride_in_b,    # = CK * L
    stride_in_ck,   # = L
    stride_in_l,    # = 1
    N_total,        # = N_out * GROUP_SIZE * KERNEL_SIZE
    GROUP_SIZE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    out[n, g, k] = in[b, group_idx * GROUP_SIZE * KERNEL_SIZE + g * KERNEL_SIZE + k, l]
    where:
      group_idx = n % num_groups
      l         = (n // num_groups) % L
      b         = n // (num_groups * L)
    Output is contiguous with linear index = n * GROUP_SIZE * KERNEL_SIZE + g * KERNEL_SIZE + k.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_total

    # Decode output linear index → (n, g, k)
    k = offsets % KERNEL_SIZE
    g = (offsets // KERNEL_SIZE) % GROUP_SIZE
    n = offsets // (KERNEL_SIZE * GROUP_SIZE)

    # Decode n → (b, l, group_idx)
    group_idx = n % num_groups
    l_b = n // num_groups
    l = l_b % L
    b = l_b // L

    # Source channel index in [B, CK, L]
    ck = group_idx * GROUP_SIZE * KERNEL_SIZE + g * KERNEL_SIZE + k

    # Flat input index
    in_idx = b * stride_in_b + ck * stride_in_ck + l * stride_in_l

    val = tl.load(in_ptr + in_idx, mask=mask)
    tl.store(out_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def fused_transpose_reshape_384_64(tmp_2):
    # tmp_2: [B, CK, L]   CK = 3456 for C=384
    B, CK, L = tmp_2.shape

    GROUP_SIZE = 64
    KERNEL_SIZE = 9
    num_groups = CK // (GROUP_SIZE * KERNEL_SIZE)   # = 6
    N_out = B * L * num_groups
    N_total = N_out * GROUP_SIZE * KERNEL_SIZE

    out = torch.empty(N_out, GROUP_SIZE, KERNEL_SIZE,
                      dtype=tmp_2.dtype, device=tmp_2.device)

    grid = lambda meta: ((N_total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    transpose_reshape_384_64_kernel[grid](
        tmp_2,
        out,
        B, L, num_groups,
        tmp_2.stride(0), tmp_2.stride(1), tmp_2.stride(2),
        N_total,
        GROUP_SIZE=GROUP_SIZE,
        KERNEL_SIZE=KERNEL_SIZE,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement interface required by the AI4C pass framework
# ---------------------------------------------------------------------------

def pattern(tmp_2):
    """
    Match the transpose + two-reshape chain.
    tmp_2 is the output of F.unfold with shape [B, 3456, L].
    """
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 384, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 64, 9])
    return tmp_5


def replacement_args(tmp_2):
    return (tmp_2,)


def replacement_func():
    return fused_transpose_reshape_384_64