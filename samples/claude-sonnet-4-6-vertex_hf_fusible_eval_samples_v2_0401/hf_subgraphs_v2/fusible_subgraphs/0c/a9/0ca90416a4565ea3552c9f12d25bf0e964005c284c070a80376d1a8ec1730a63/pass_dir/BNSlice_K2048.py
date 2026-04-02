import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['C', 'HW'],
)
@triton.jit
def _bn_affine_kernel_k2048(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    nc = tl.program_id(0)
    hw_block = tl.program_id(1)

    c = nc % C

    mean = tl.load(mean_ptr + c).to(tl.float32)
    var = tl.load(var_ptr + c).to(tl.float32)
    w = tl.load(weight_ptr + c).to(tl.float32)
    b = tl.load(bias_ptr + c).to(tl.float32)

    inv_std = w * tl.rsqrt(var + eps)
    bias_val = b - mean * inv_std

    hw_start = hw_block * BLOCK_SIZE
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    mask = hw_offsets < HW

    base = nc * HW
    x = tl.load(x_ptr + base + hw_offsets, mask=mask, other=0.0).to(tl.float32)
    out = x * inv_std + bias_val
    tl.store(out_ptr + base + hw_offsets, out, mask=mask)


@torch.fx.wrap
def _bn_slice_K2048(in_0, in_1, in_2, in_3, in_4, in_5):
    device = in_4.device

    mean = in_0.to(device=device)
    var = in_1.to(device=device)
    weight = in_3.to(device=device)
    bias = in_2.to(device=device)

    x = in_4.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)

    grid = lambda meta: (N * C, triton.cdiv(HW, meta['BLOCK_SIZE']))
    _bn_affine_kernel_k2048[grid](x, mean, var, weight, bias, out, C, HW, 0.001)

    tmp_4 = in_5[:, 2048:, :, :]
    return (out, tmp_4)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = in_5[(slice(None, None, None), slice(2048, None, None), slice(None, None, None), slice(None, None, None))]
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return (tmp_5, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return _bn_slice_K2048