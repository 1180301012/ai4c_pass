import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["C", "OUT_DTYPE_ID"],
)
@triton.jit
def _ln_tail_kernel(
    conv_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    tmp7_ptr,
    tmp10_ptr,
    C,
    H,
    W,
    tmp7_stride_n,
    tmp7_stride_hw,
    tmp7_stride_c,
    tmp10_stride_hw,
    tmp10_stride_n,
    tmp10_stride_c,
    eps,
    OUT_DTYPE_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    hw_size = H * W
    n_idx = pid // hw_size
    hw_idx = pid % hw_size

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < C

    base = n_idx * C * hw_size + offs * hw_size + hw_idx
    conv = tl.load(conv_ptr + base, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
    x = conv + residual

    if OUT_DTYPE_ID == 0:
        x_out = x.to(tl.float32)
    else:
        x_out = x.to(tl.bfloat16)

    tmp7_ptrs = tmp7_ptr + n_idx * tmp7_stride_n + hw_idx * tmp7_stride_hw + offs * tmp7_stride_c
    tl.store(tmp7_ptrs, x_out, mask=mask)

    mean = tl.sum(x, axis=0) / C
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / C
    rstd = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x_centered * rstd
    z = y * weight + bias

    if OUT_DTYPE_ID == 0:
        z_out = z.to(tl.float32)
    else:
        z_out = z.to(tl.bfloat16)

    tmp10_ptrs = tmp10_ptr + hw_idx * tmp10_stride_hw + n_idx * tmp10_stride_n + offs * tmp10_stride_c
    tl.store(tmp10_ptrs, z_out, mask=mask)


@torch.fx.wrap
def fused_residual_ln_tail(conv_out, residual, weight, bias):
    n, c, h, w = conv_out.shape
    hw = h * w

    tmp7 = torch.empty((n, hw, c), device=conv_out.device, dtype=conv_out.dtype)
    tmp10 = torch.empty((hw, n, c), device=conv_out.device, dtype=conv_out.dtype)

    if conv_out.dtype == torch.float32:
        out_dtype_id = 0
    elif conv_out.dtype == torch.bfloat16:
        out_dtype_id = 1
    else:
        raise RuntimeError(f"unsupported dtype: {conv_out.dtype}")

    grid = (n * hw,)
    _ln_tail_kernel[grid](
        conv_out,
        residual,
        weight,
        bias,
        tmp7,
        tmp10,
        c,
        h,
        w,
        tmp7.stride(0),
        tmp7.stride(1),
        tmp7.stride(2),
        tmp10.stride(0),
        tmp10.stride(1),
        tmp10.stride(2),
        1e-5,
        OUT_DTYPE_ID=out_dtype_id,
    )

    tmp9 = tmp10
    return tmp7, tmp10, tmp9


def replacement_dispatch(in_0, in_1, in_2, in_3):
    outs = fused_residual_ln_tail(in_0, in_1, in_2, in_3)
    return outs[0], outs[1], outs[2]