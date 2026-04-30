import torch
import triton
import triton.language as tl


def pattern(in_1):
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0.mean((2, 3))
    tmp_4 = tmp_1.view(1, 1, -1)
    return tmp_0, tmp_4


def replacement_args(in_1):
    return (in_1,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
    ],
    key=['n_hw'],
)
@triton.jit
def fused_silu_mean_kernel(
    input_ptr,
    silu_ptr,
    mean_ptr,
    n_channels,
    n_hw,
    inv_hw,
    stride_c,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)

    if c >= n_channels:
        return

    in_base = input_ptr + c * stride_c
    out_base = silu_ptr + c * stride_c

    acc = 0.0

    for hw_start in range(0, n_hw, BLOCK_HW):
        offsets = hw_start + tl.arange(0, BLOCK_HW)
        mask = offsets < n_hw

        x = tl.load(in_base + offsets, mask=mask, other=0.0).to(tl.float32)
        silu_val = x * tl.sigmoid(x)
        tl.store(out_base + offsets, silu_val, mask=mask)
        acc += tl.sum(silu_val)

    mean_val = acc * inv_hw
    tl.store(mean_ptr + c, mean_val)


@torch.fx.wrap
def fused_silu_mean(in_1):
    C = in_1.shape[1]
    H = in_1.shape[2]
    W = in_1.shape[3]
    HW = H * W
    inv_hw = 1.0 / HW
    stride_c = in_1.stride(1)

    silu_out = torch.empty_like(in_1)
    mean_out = torch.empty((1, 1, C), dtype=in_1.dtype, device=in_1.device)

    grid = (C,)

    fused_silu_mean_kernel[grid](
        in_1, silu_out, mean_out,
        C, HW, inv_hw, stride_c,
    )

    return silu_out, mean_out


def replacement_func():
    return fused_silu_mean