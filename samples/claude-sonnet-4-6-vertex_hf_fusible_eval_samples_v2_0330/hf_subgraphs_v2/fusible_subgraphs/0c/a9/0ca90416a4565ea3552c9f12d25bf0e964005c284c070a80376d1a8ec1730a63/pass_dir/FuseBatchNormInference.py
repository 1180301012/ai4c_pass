import torch
import triton
import triton.language as tl


def pattern(a, b, c, d, e):
    """
    Match batch_norm inference call.
    a = running_mean (in_0)
    b = running_var  (in_1)
    c = bias         (in_2)
    d = weight       (in_3)
    e = input        (in_4)

    The model calls:
      torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    which maps to:
      torch.nn.functional.batch_norm(e,    a,    b,    d,    c,    False, 0.1, 0.001)
    """
    result = torch.nn.functional.batch_norm(e, a, b, d, c, False, 0.1, 0.001)
    return result


def replacement_args(a, b, c, d, e):
    return (a, b, c, d, e)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def _bn_inference_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """
    2-D grid: axis-0 is (N*C), axis-1 tiles the HW dimension.
    Each program loads the per-channel stats once, then normalises a
    BLOCK_HW-wide strip of the spatial plane.
    """
    nc_idx      = tl.program_id(0)
    hw_block_id = tl.program_id(1)
    c_idx       = nc_idx % C

    # ---- per-channel stats (computed in fp32 for numerical stability) ----
    eps = 1e-3

    mean = tl.load(mean_ptr   + c_idx).to(tl.float32)
    var  = tl.load(var_ptr    + c_idx).to(tl.float32)
    w    = tl.load(weight_ptr + c_idx).to(tl.float32)
    b    = tl.load(bias_ptr   + c_idx).to(tl.float32)

    inv_std = tl.rsqrt(var + eps)
    scale   = w * inv_std
    shift   = b - mean * scale

    # ---- spatial strip ----
    hw_start = hw_block_id * BLOCK_HW
    offsets  = hw_start + tl.arange(0, BLOCK_HW)
    mask     = offsets < HW

    base = nc_idx * HW
    x_val   = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
    out_f32 = x_val.to(tl.float32) * scale + shift
    tl.store(out_ptr + base + offsets, out_f32.to(x_val.dtype), mask=mask)


@torch.fx.wrap
def triton_batch_norm_inference(a, b, c, d, e):
    """
    Optimised batch-norm inference.
    a=running_mean, b=running_var, c=bias, d=weight, e=input
    """
    running_mean = a
    running_var  = b
    bias         = c
    weight       = d
    x            = e

    # Some graphs have BN buffers on CPU — move them to the input device.
    device = x.device
    if running_mean.device != device:
        running_mean = running_mean.to(device)
    if running_var.device != device:
        running_var = running_var.to(device)
    if weight.device != device:
        weight = weight.to(device)
    if bias.device != device:
        bias = bias.to(device)

    N, C, H, W = x.shape
    HW  = H * W
    out = torch.empty_like(x)

    grid = lambda meta: (N * C, triton.cdiv(HW, meta['BLOCK_HW']))

    _bn_inference_kernel[grid](
        x, running_mean, running_var, weight, bias, out,
        C, HW,
    )

    return out


def replacement_func():
    return triton_batch_norm_inference