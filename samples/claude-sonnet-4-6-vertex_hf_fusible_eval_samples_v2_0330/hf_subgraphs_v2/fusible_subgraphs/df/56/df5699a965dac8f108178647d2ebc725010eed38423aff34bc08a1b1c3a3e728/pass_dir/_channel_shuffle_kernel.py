import torch
import triton
import triton.language as tl


# ── Triton kernel for channel shuffle + chunk ──────────────────────────────
# Inputs  a, b : [N, C, H, W]  (both same shape, C channels each)
# Outputs out0, out1 : [N, C, H, W]
#
# out0[n, 2j,   hw] = a[n, j,        hw]   for j in [0, C_half)
# out0[n, 2j+1, hw] = b[n, j,        hw]   for j in [0, C_half)
# out1[n, 2j,   hw] = a[n, j+C_half, hw]   for j in [0, C_half)
# out1[n, 2j+1, hw] = b[n, j+C_half, hw]   for j in [0, C_half)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N', 'C_HALF', 'HW'],
)
@triton.jit
def channel_shuffle_kernel(
    a_ptr, b_ptr, out0_ptr, out1_ptr,
    N, C_HALF, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C_HALF * HW
    mask = offsets < total

    # Decompose flat index: [N, C_half, HW] order
    hw  = offsets % HW
    j   = (offsets // HW) % C_HALF
    n   = offsets // (HW * C_HALF)

    C = 2 * C_HALF
    # Source indices  (a, b both stride [N, C, HW])
    a_lo = n * C * HW + j * HW + hw
    a_hi = n * C * HW + (j + C_HALF) * HW + hw

    a_lower = tl.load(a_ptr + a_lo, mask=mask)
    b_lower = tl.load(b_ptr + a_lo, mask=mask)   # b has same shape/stride as a
    a_upper = tl.load(a_ptr + a_hi, mask=mask)
    b_upper = tl.load(b_ptr + a_hi, mask=mask)

    # Destination indices in out0, out1  (both stride [N, C, HW])
    out0_even = n * C * HW + (2 * j)     * HW + hw
    out0_odd  = n * C * HW + (2 * j + 1) * HW + hw

    tl.store(out0_ptr + out0_even, a_lower, mask=mask)
    tl.store(out0_ptr + out0_odd,  b_lower, mask=mask)
    tl.store(out1_ptr + out0_even, a_upper, mask=mask)
    tl.store(out1_ptr + out0_odd,  b_upper, mask=mask)