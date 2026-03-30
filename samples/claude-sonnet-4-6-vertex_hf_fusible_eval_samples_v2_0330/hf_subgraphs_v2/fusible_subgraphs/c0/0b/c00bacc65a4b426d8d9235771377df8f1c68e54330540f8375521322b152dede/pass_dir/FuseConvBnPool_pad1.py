import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Matches: conv2d(pad=1,1) -> batch_norm (inference) + avg_pool2d (independent)
    in_0: BN running_mean, in_1: BN running_var, in_2: BN bias, in_3: BN weight
    in_4: conv weight, in_5: conv input, in_6: pool input
    """
    conv2d = torch.conv2d(in_5, in_4, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.avg_pool2d(in_6, 2, 2, 0, True, False, None)
    return (tmp_7, tmp_6)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _bn_inference_kernel_pad1(
    x_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    C,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """
    Grid: (N*C, ceil(HW / BLOCK_HW))
    Each program handles one (n,c) slice and BLOCK_HW spatial positions.
    Applies: out = x * scale[c] + shift[c]
    """
    nc_pid = tl.program_id(0)
    hw_pid = tl.program_id(1)

    c = nc_pid % C

    hw_offsets = hw_pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = hw_offsets < HW

    base = nc_pid * HW

    x = tl.load(x_ptr + base + hw_offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + c)
    shift = tl.load(shift_ptr + c)

    out = x * scale + shift
    tl.store(out_ptr + base + hw_offsets, out, mask=mask)


@torch.fx.wrap
def _fused_conv_bn_pool_pad1(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    """
    Replacement for: conv2d(pad=1) + batch_norm + avg_pool2d
    - in_5: conv input (CUDA)
    - in_6: pool input (CUDA)
    - in_0: BN running_mean, in_1: running_var, in_2: bias, in_3: weight
    - in_4: conv weight
    """
    device = in_5.device
    dtype = in_5.dtype

    # Move all params to same device (some may be on CPU in test configs)
    running_mean = in_0.to(device=device, dtype=torch.float32)
    running_var  = in_1.to(device=device, dtype=torch.float32)
    bn_gamma     = in_3.to(device=device, dtype=torch.float32)  # weight
    bn_beta      = in_2.to(device=device, dtype=torch.float32)  # bias
    conv_weight  = in_4.to(device=device)

    # Run convolution (cuDNN optimized)
    conv_out = torch.conv2d(in_5, conv_weight, None, (1, 1), (1, 1), (1, 1), 1)

    # Pre-compute BN affine params in float32, then cast to activation dtype
    eps = 1e-5
    scale_f32 = bn_gamma / torch.sqrt(running_var + eps)
    shift_f32 = bn_beta - running_mean * scale_f32
    scale = scale_f32.to(dtype=dtype)
    shift = shift_f32.to(dtype=dtype)

    # Apply BN element-wise with Triton kernel
    conv_out = conv_out.contiguous()
    N, C, H, W = conv_out.shape
    HW = H * W
    bn_out = torch.empty_like(conv_out)

    grid = lambda meta: (N * C, triton.cdiv(HW, meta['BLOCK_HW']))
    _bn_inference_kernel_pad1[grid](
        conv_out, scale, shift, bn_out,
        C, HW,
    )

    # Average pooling (independent of conv+BN branch)
    pool_out = torch.nn.functional.avg_pool2d(in_6, 2, 2, 0, True, False, None)

    return (pool_out, bn_out)


def replacement_func():
    return _fused_conv_bn_pool_pad1