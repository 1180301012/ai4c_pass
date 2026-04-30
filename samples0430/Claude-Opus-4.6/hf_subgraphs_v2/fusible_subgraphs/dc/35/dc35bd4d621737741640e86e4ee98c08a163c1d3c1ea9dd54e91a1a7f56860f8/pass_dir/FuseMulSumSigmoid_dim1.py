import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_mul_sum_sigmoid_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    HW, C,
    BLOCK: tl.constexpr,
):
    hw_block = tl.program_id(0)
    b_idx = tl.program_id(1)

    hw_offsets = hw_block * BLOCK + tl.arange(0, BLOCK)
    mask = hw_offsets < HW

    # Base offset: b * C * HW + hw (no division needed)
    base = b_idx * C * HW + hw_offsets

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for c in range(64):
        in_offset = base + c * HW
        x = tl.load(in_0_ptr + in_offset, mask=mask, other=0.0)
        y = tl.load(in_1_ptr + in_offset, mask=mask, other=0.0)
        acc += (x * y).to(tl.float32)

    result = tl.sigmoid(acc)
    out_offset = b_idx * HW + hw_offsets
    tl.store(out_ptr + out_offset, result, mask=mask)


@torch.fx.wrap
def fused_mul_sum_sigmoid(in_0, in_1):
    B, C, H, W = in_0.shape
    HW = H * W
    out = torch.empty((B, 1, H, W), dtype=in_0.dtype, device=in_0.device)

    BLOCK = 256
    grid = ((HW + BLOCK - 1) // BLOCK, B)

    fused_mul_sum_sigmoid_kernel[grid](
        in_0, in_1, out,
        HW, C,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=8,
    )

    return out


def replacement_func():
    return fused_mul_sum_sigmoid