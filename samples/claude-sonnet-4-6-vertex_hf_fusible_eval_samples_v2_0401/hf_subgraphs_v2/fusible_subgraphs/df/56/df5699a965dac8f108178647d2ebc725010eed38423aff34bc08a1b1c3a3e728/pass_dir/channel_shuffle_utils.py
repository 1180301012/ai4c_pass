"""
Shared Triton kernels for channel shuffle + split operations.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def channel_shuffle_out0_kernel(
    A_ptr, B_ptr, Out0_ptr,
    N, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program handles one (n, i) pair for output 0.
    Out0[:, i, :, :] = A[:, i//2, :, :] if i%2==0 else B[:, i//2, :, :]
    Grid: (N * C, ceil(HW / BLOCK_SIZE))
    """
    pid = tl.program_id(0)
    hw_block = tl.program_id(1)

    n = pid // C
    i = pid % C          # output channel in [0, C)
    c_src = i // 2       # source channel in [0, C//2)
    use_B = (i % 2) == 1

    hw_start = hw_block * BLOCK_SIZE
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    mask = hw_offsets < HW

    src_offset = n * C * HW + c_src * HW + hw_offsets
    a_vals = tl.load(A_ptr + src_offset, mask=mask, other=0.0)
    b_vals = tl.load(B_ptr + src_offset, mask=mask, other=0.0)
    vals = tl.where(use_B, b_vals, a_vals)

    out_offset = n * C * HW + i * HW + hw_offsets
    tl.store(Out0_ptr + out_offset, vals, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def channel_shuffle_out1_kernel(
    A_ptr, B_ptr, Out1_ptr,
    N, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program handles one (n, i) pair for output 1.
    Out1[:, i, :, :] = A[:, C//2 + i//2, :, :] if i%2==0 else B[:, C//2 + i//2, :, :]
    Grid: (N * C, ceil(HW / BLOCK_SIZE))
    """
    pid = tl.program_id(0)
    hw_block = tl.program_id(1)

    n = pid // C
    i = pid % C           # output channel in [0, C)
    C_HALF = C // 2
    c_src = C_HALF + i // 2   # source channel in [C//2, C)
    use_B = (i % 2) == 1

    hw_start = hw_block * BLOCK_SIZE
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    mask = hw_offsets < HW

    src_offset = n * C * HW + c_src * HW + hw_offsets
    a_vals = tl.load(A_ptr + src_offset, mask=mask, other=0.0)
    b_vals = tl.load(B_ptr + src_offset, mask=mask, other=0.0)
    vals = tl.where(use_B, b_vals, a_vals)

    out_offset = n * C * HW + i * HW + hw_offsets
    tl.store(Out1_ptr + out_offset, vals, mask=mask)


def launch_channel_shuffle(A, B, C_fixed, H_fixed, W_fixed):
    """
    Launches channel shuffle split kernels for inputs A and B.
    A and B both have shape [N, C_fixed, H_fixed, W_fixed].
    Returns (Out0, Out1) each of shape [N, C_fixed, H_fixed, W_fixed].
    """
    N = A.shape[0]
    C = C_fixed
    HW = H_fixed * W_fixed

    Out0 = torch.empty_like(A)
    Out1 = torch.empty_like(A)

    grid = lambda meta: (N * C, triton.cdiv(HW, meta['BLOCK_SIZE']))

    channel_shuffle_out0_kernel[grid](
        A, B, Out0,
        N, C, HW,
    )
    channel_shuffle_out1_kernel[grid](
        A, B, Out1,
        N, C, HW,
    )

    return Out0, Out1