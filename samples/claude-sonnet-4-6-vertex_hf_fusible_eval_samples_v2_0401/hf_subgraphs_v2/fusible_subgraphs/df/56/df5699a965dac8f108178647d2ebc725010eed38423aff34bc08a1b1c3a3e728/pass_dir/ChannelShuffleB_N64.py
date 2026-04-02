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
def _csb_out0_kernel(A_ptr, B_ptr, Out0_ptr, N, C, HW, BLOCK_SIZE: tl.constexpr):
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
def _csb_out1_kernel(A_ptr, B_ptr, Out1_ptr, N, C, HW, BLOCK_SIZE: tl.constexpr):
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
def channel_shuffle_B(A, B):
    N = A.shape[0]
    C = 40
    HW = 32 * 24
    Out0 = torch.empty_like(A)
    Out1 = torch.empty_like(A)
    grid = lambda meta: (N * C, triton.cdiv(HW, meta['BLOCK_SIZE']))
    _csb_out0_kernel[grid](A, B, Out0, N, C, HW)
    _csb_out1_kernel[grid](A, B, Out1, N, C, HW)
    return (Out0, Out1)


def pattern(in_3, tmp_4):
    tmp_6 = torch.cat([in_3, tmp_4], dim=1)
    tmp_11 = tmp_6.view(64, 2, 40, 32, 24)
    tmp_12 = torch.transpose(tmp_11, 1, 2)
    tmp_13 = tmp_12.contiguous()
    tmp_14 = tmp_13.view(64, 80, 32, 24)
    chunk_1 = tmp_14.chunk(2, dim=1)
    tmp_19 = chunk_1[0]
    tmp_20 = chunk_1[1]
    return (tmp_19, tmp_20)


def replacement_args(in_3, tmp_4):
    return (in_3, tmp_4)


def replacement_func():
    return channel_shuffle_B