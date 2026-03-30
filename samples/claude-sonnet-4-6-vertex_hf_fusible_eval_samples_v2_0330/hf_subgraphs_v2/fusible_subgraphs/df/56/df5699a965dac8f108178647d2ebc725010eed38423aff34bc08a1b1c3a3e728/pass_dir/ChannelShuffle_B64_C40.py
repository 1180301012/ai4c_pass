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
def _cs_kernel_B64_C40(
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
def channel_shuffle_64_C40(in_3, tmp_4):
    N, C, H, W = 64, 40, 32, 24
    C_HALF, HW = C // 2, H * W
    out0 = torch.empty(N, C, H, W, dtype=in_3.dtype, device=in_3.device)
    out1 = torch.empty(N, C, H, W, dtype=in_3.dtype, device=in_3.device)
    total = N * C_HALF * HW
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    _cs_kernel_B64_C40[grid](in_3, tmp_4, out0, out1, N, C_HALF, HW)
    return out0, out1


def pattern(in_3, tmp_4):
    tmp_6 = torch.cat([in_3, tmp_4], dim=1)
    tmp_11 = tmp_6.view(64, 2, 40, 32, 24)
    tmp_12 = torch.transpose(tmp_11, 1, 2)
    tmp_13 = tmp_12.contiguous()
    tmp_14 = tmp_13.view(64, 80, 32, 24)
    chunk_1 = tmp_14.chunk(2, dim=1)
    tmp_19 = chunk_1[0]
    tmp_20 = chunk_1[1]
    return tmp_19, tmp_20


def replacement_args(in_3, tmp_4):
    return (in_3, tmp_4)


def replacement_func():
    return channel_shuffle_64_C40