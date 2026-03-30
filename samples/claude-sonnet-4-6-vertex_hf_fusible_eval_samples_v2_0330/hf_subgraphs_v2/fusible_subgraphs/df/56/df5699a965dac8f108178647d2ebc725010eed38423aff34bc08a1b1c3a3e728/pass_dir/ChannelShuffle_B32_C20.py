import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N', 'C_HALF', 'HW'],
)
@triton.jit
def _cs_kernel_B32_C20(
    a_ptr, b_ptr, out0_ptr, out1_ptr,
    N, C_HALF, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C_HALF * HW
    mask = offsets < total
    hw = offsets % HW
    j  = (offsets // HW) % C_HALF
    n  = offsets // (HW * C_HALF)
    C = 2 * C_HALF
    a_lo = n * C * HW + j * HW + hw
    a_hi = n * C * HW + (j + C_HALF) * HW + hw
    a_lower = tl.load(a_ptr + a_lo, mask=mask)
    b_lower = tl.load(b_ptr + a_lo, mask=mask)
    a_upper = tl.load(a_ptr + a_hi, mask=mask)
    b_upper = tl.load(b_ptr + a_hi, mask=mask)
    out_even = n * C * HW + (2 * j)     * HW + hw
    out_odd  = n * C * HW + (2 * j + 1) * HW + hw
    tl.store(out0_ptr + out_even, a_lower, mask=mask)
    tl.store(out0_ptr + out_odd,  b_lower, mask=mask)
    tl.store(out1_ptr + out_even, a_upper, mask=mask)
    tl.store(out1_ptr + out_odd,  b_upper, mask=mask)


@torch.fx.wrap
def channel_shuffle_32_C20(in_2, in_4):
    N, C, H, W = 32, 20, 64, 48
    C_HALF, HW = C // 2, H * W
    out0 = torch.empty(N, C, H, W, dtype=in_2.dtype, device=in_2.device)
    out1 = torch.empty(N, C, H, W, dtype=in_2.dtype, device=in_2.device)
    total = N * C_HALF * HW
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    _cs_kernel_B32_C20[grid](in_2, in_4, out0, out1, N, C_HALF, HW)
    return out0, out1


def pattern(in_2, in_4):
    tmp_5 = torch.cat([in_2, in_4], dim=1)
    tmp_7 = tmp_5.view(32, 2, 20, 64, 48)
    tmp_8 = torch.transpose(tmp_7, 1, 2)
    tmp_9 = tmp_8.contiguous()
    tmp_10 = tmp_9.view(32, 40, 64, 48)
    chunk = tmp_10.chunk(2, dim=1)
    tmp_16 = chunk[0]
    tmp_17 = chunk[1]
    return tmp_16, tmp_17


def replacement_args(in_2, in_4):
    return (in_2, in_4)


def replacement_func():
    return channel_shuffle_32_C20