import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
    ],
    key=['N', 'C_OUT', 'HW'],
)
@triton.jit
def fused_conv_sigmoid_mul_kernel(
    in6_ptr,    # [N, C_in, 1, 1]
    weight_ptr, # [C_OUT, C_in, 1, 1]
    bias_ptr,   # [C_OUT]
    in5_ptr,    # [N, C_OUT, H, W]
    out_ptr,    # [N, C_OUT, H, W]
    N, C_in, C_OUT, HW,
    BLOCK_HW: tl.constexpr,
):
    # Grid: (N * C_OUT, ceil(HW / BLOCK_HW))
    nc_idx = tl.program_id(0)
    hw_block = tl.program_id(1)

    n_idx = nc_idx // C_OUT
    c_idx = nc_idx % C_OUT

    # Compute linear output for this (n, c): bias + sum(in6[n,:] * weight[c,:])
    val = tl.load(bias_ptr + c_idx).to(tl.float32)
    for k in range(C_in):
        in6_val = tl.load(in6_ptr + n_idx * C_in + k).to(tl.float32)
        w_val = tl.load(weight_ptr + c_idx * C_in + k).to(tl.float32)
        val = val + in6_val * w_val

    # Sigmoid
    val = 1.0 / (1.0 + tl.exp(-val))

    # Apply to all HW positions
    hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = hw_offsets < HW

    base = (n_idx * C_OUT + c_idx) * HW
    in5_vals = tl.load(in5_ptr + base + hw_offsets, mask=mask)
    result = in5_vals.to(tl.float32) * val
    tl.store(out_ptr + base + hw_offsets, result.to(in5_vals.dtype), mask=mask)


@torch.fx.wrap
def fused_conv_sigmoid_mul(in_6, in_1, in_0, in_5):
    N = in_6.shape[0]
    C_in = in_6.shape[1]   # 10
    C_OUT = in_1.shape[0]  # 40
    H = in_5.shape[2]      # 32
    W = in_5.shape[3]      # 24
    HW = H * W             # 768

    out = torch.empty_like(in_5)

    grid = lambda meta: (N * C_OUT, triton.cdiv(HW, meta['BLOCK_HW']))
    fused_conv_sigmoid_mul_kernel[grid](
        in_6.contiguous(), in_1.contiguous(), in_0.contiguous(), in_5.contiguous(), out,
        N, C_in, C_OUT, HW,
    )
    return out


def pattern(in_6, in_1, in_0, in_5):
    conv2d = torch.conv2d(in_6, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = in_5 * tmp_3
    return tmp_4


def replacement_args(in_6, in_1, in_0, in_5):
    return (in_6, in_1, in_0, in_5)


def replacement_func():
    return fused_conv_sigmoid_mul