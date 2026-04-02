import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
    ],
    key=['N', 'C_out', 'H', 'W'],
)
@triton.jit
def nchw_to_nhwc_sigmoid_kernel_16_9(
    in_ptr,
    out_ptr,
    N, C_out, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    total = N * C_out * H * W
    mask = offsets < total

    c = offsets % C_out
    tmp = offsets // C_out
    w_idx = tmp % W
    tmp = tmp // W
    h_idx = tmp % H
    n_idx = tmp // H

    in_idx = n_idx * (C_out * H * W) + c * (H * W) + h_idx * W + w_idx

    x = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    result = 1.0 / (1.0 + tl.exp(-x_f32))
    result = result.to(x.dtype)
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_permute_sigmoid_16_9(x):
    N, C_out, H, W = x.shape
    total = N * C_out * H * W

    out_flat = torch.empty(total, dtype=x.dtype, device=x.device)

    grid = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    nchw_to_nhwc_sigmoid_kernel_16_9[grid](
        x, out_flat,
        N, C_out, H, W,
    )

    return (out_flat.reshape(16, -1, 9),)


def pattern(x):
    tmp_3 = x.permute(0, 2, 3, 1)
    tmp_4 = tmp_3.reshape(16, -1, 9)
    tmp_5 = torch.nn.functional.sigmoid(tmp_4)
    return (tmp_5,)


def replacement_args(x):
    return (x,)


def replacement_func():
    return fused_permute_sigmoid_16_9