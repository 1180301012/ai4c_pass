import torch
import triton
import triton.language as tl


def pattern(in_0, tmp_0):
    tmp_1 = in_0 * tmp_0
    tmp_2 = torch.sum(tmp_1, dim=1)
    return (tmp_2,)


def replacement_args(in_0, tmp_0):
    return (in_0, tmp_0)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 16, "BLOCK_HW": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 32, "BLOCK_HW": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 32, "BLOCK_HW": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_C": 32, "BLOCK_HW": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_C": 64, "BLOCK_HW": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 64, "BLOCK_HW": 128}, num_warps=8, num_stages=1),
    ],
    key=["HW"],
)
@triton.jit
def _weighted_sum_dim1_size2_kernel(
    in0_ptr,
    w_ptr,
    out_ptr,
    HW,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offsets < HW
    mask = hw_mask[None, :]

    w0 = tl.load(w_ptr + c_offsets).to(tl.float32)[:, None]
    w1 = tl.load(w_ptr + 256 + c_offsets).to(tl.float32)[:, None]

    base = c_offsets[:, None] * HW + hw_offsets[None, :]
    x0 = tl.load(in0_ptr + base, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(in0_ptr + 256 * HW + base, mask=mask, other=0.0).to(tl.float32)
    out_val = x0 * w0 + x1 * w1
    tl.store(out_ptr + base, out_val, mask=mask)


@torch.fx.wrap
def fused_weighted_sum_dim1_size2(in_0, tmp_0):
    H = in_0.shape[3]
    W = in_0.shape[4]
    HW = H * W

    out = torch.empty((1, 256, H, W), device=in_0.device, dtype=in_0.dtype)

    grid = lambda META: (triton.cdiv(HW, META["BLOCK_HW"]), 256 // META["BLOCK_C"])
    _weighted_sum_dim1_size2_kernel[grid](
        in_0,
        tmp_0,
        out,
        HW,
    )
    return out


def replacement_func():
    return fused_weighted_sum_dim1_size2