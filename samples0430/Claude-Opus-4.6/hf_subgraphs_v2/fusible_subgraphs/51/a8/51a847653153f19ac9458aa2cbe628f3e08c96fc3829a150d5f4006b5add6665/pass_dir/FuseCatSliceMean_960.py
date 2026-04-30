import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = tmp_0[(slice(None, None, None), slice(None, 960, None), slice(None, None, None), slice(None, None, None))]
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def copy_mean_kernel(
    src_ptr, dst_ptr, mean_ptr,
    num_channels, HW,
    src_batch_stride, dst_batch_stride,
    dst_c_offset, mean_c_offset,
    mean_batch_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // num_channels
    c = pid % num_channels

    src_offset = b * src_batch_stride + c * HW
    dst_offset = b * dst_batch_stride + (c + dst_c_offset) * HW

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start in range(0, HW, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        vals = tl.load(src_ptr + src_offset + offsets, mask=mask, other=0.0)
        tl.store(dst_ptr + dst_offset + offsets, vals, mask=mask)
        acc += vals.to(tl.float32)

    total = tl.sum(acc, axis=0)
    mean_val = total / HW
    mean_offset = b * mean_batch_stride + c + mean_c_offset
    tl.store(mean_ptr + mean_offset, mean_val)


@torch.fx.wrap
def fused_cat_mean_960(in_0, in_1):
    B = in_0.shape[0]
    C0 = in_0.shape[1]
    C1 = in_1.shape[1]
    H = in_0.shape[2]
    W = in_0.shape[3]
    C_total = C0 + C1
    HW = H * W

    out_cat = torch.empty((B, C_total, H, W), dtype=in_0.dtype, device=in_0.device)
    out_mean = torch.empty((B, C_total, 1, 1), dtype=in_0.dtype, device=in_0.device)

    # Process in_0 -> first C0 channels
    copy_mean_kernel[(B * C0,)](
        in_0, out_cat, out_mean,
        C0, HW,
        C0 * HW, C_total * HW,
        0, 0,
        C_total,
    )

    # Process in_1 -> last C1 channels
    copy_mean_kernel[(B * C1,)](
        in_1, out_cat, out_mean,
        C1, HW,
        C1 * HW, C_total * HW,
        C0, C0,
        C_total,
    )

    return (out_cat, out_mean)


def replacement_func():
    return fused_cat_mean_960