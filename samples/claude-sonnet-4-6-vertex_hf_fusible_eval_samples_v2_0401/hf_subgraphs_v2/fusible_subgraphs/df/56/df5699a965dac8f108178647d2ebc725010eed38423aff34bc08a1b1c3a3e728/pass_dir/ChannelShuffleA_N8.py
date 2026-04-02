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
def _csa_out0_kernel(A_ptr, B_ptr, Out0_ptr, N, C, HW, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    hw_block = tl.program_id(1)
    n = pid // C
    i = pid % C
    c_src = i // 2
    use_B = (i % 2) == 1
    hw_start = hw_block * BLOCK_SIZE
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    mask = hw_offsets < HW
    src_offset = n * C * HW + c_src * HW + hw_offsets
    a_vals = tl.load(A_ptr + src_offset, mask=mask, other=0.0)
    b_vals = tl.load(B_ptr + src_offset, mask=mask, other=0.0)
    vals = tl.where(use_B, b_vals, a_vals)
    tl.store(Out0_ptr + n * C * HW + i * HW + hw_offsets, vals, mask=mask)


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
def _csa_out1_kernel(A_ptr, B_ptr, Out1_ptr, N, C, HW, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    hw_block = tl.program_id(1)
    n = pid // C
    i = pid % C
    C_HALF = C // 2
    c_src = C_HALF + i // 2
    use_B = (i % 2) == 1
    hw_start = hw_block * BLOCK_SIZE
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    mask = hw_offsets < HW
    src_offset = n * C * HW + c_src * HW + hw_offsets
    a_vals = tl.load(A_ptr + src_offset, mask=mask, other=0.0)
    b_vals = tl.load(B_ptr + src_offset, mask=mask, other=0.0)
    vals = tl.where(use_B, b_vals, a_vals)
    tl.store(Out1_ptr + n * C * HW + i * HW + hw_offsets, vals, mask=mask)


@torch.fx.wrap
def channel_shuffle_A(A, B):
    N = A.shape[0]
    C = 20
    HW = 64 * 48
    Out0 = torch.empty_like(A)
    Out1 = torch.empty_like(A)
    grid = lambda meta: (N * C, triton.cdiv(HW, meta['BLOCK_SIZE']))
    _csa_out0_kernel[grid](A, B, Out0, N, C, HW)
    _csa_out1_kernel[grid](A, B, Out1, N, C, HW)
    return (Out0, Out1)


def pattern(in_2, in_4):
    tmp_5 = torch.cat([in_2, in_4], dim=1)
    tmp_7 = tmp_5.view(8, 2, 20, 64, 48)
    tmp_8 = torch.transpose(tmp_7, 1, 2)
    tmp_9 = tmp_8.contiguous()
    tmp_10 = tmp_9.view(8, 40, 64, 48)
    chunk = tmp_10.chunk(2, dim=1)
    tmp_16 = chunk[0]
    tmp_17 = chunk[1]
    return (tmp_16, tmp_17)


def replacement_args(in_2, in_4):
    return (in_2, in_4)


def replacement_func():
    return channel_shuffle_A