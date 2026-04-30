import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 16, "BLOCK_C": 64}, num_warps=4),
        triton.Config({"BLOCK_HW": 32, "BLOCK_C": 64}, num_warps=4),
        triton.Config({"BLOCK_HW": 64, "BLOCK_C": 64}, num_warps=4),
        triton.Config({"BLOCK_HW": 16, "BLOCK_C": 128}, num_warps=4),
        triton.Config({"BLOCK_HW": 32, "BLOCK_C": 128}, num_warps=4),
        triton.Config({"BLOCK_HW": 64, "BLOCK_C": 128}, num_warps=8),
        triton.Config({"BLOCK_HW": 16, "BLOCK_C": 256}, num_warps=8),
        triton.Config({"BLOCK_HW": 32, "BLOCK_C": 256}, num_warps=8),
    ],
    key=["C", "HW"],
)
@triton.jit
def _silu_gap_flatten_kernel(
    in_ptr,
    out_ptr,
    C,
    HW,
    W,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    out_stride_n,
    out_stride_c,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    hw_start = 0
    while hw_start < HW:
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW

        h_offsets = hw_offsets // W
        w_offsets = hw_offsets % W

        ptrs = (
            in_ptr
            + pid_n * stride_n
            + c_offsets[:, None] * stride_c
            + h_offsets[None, :] * stride_h
            + w_offsets[None, :] * stride_w
        )
        mask = c_mask[:, None] & hw_mask[None, :]

        x = tl.load(ptrs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        silu_x = x_f32 / (1.0 + tl.exp(-x_f32))
        acc += tl.sum(silu_x, axis=1)

        hw_start += BLOCK_HW

    out_vals = acc / HW
    out_ptrs = out_ptr + pid_n * out_stride_n + c_offsets * out_stride_c
    tl.store(out_ptrs, out_vals, mask=c_mask)


@torch.fx.wrap
def fused_silu_adaptive_avg_pool2d_flatten(in_0):
    N = in_0.shape[0]
    C = in_0.shape[1]
    H = in_0.shape[2]
    W = in_0.shape[3]
    HW = H * W

    out = torch.empty((N, C), device=in_0.device, dtype=in_0.dtype)

    grid = lambda meta: (N, triton.cdiv(C, meta["BLOCK_C"]))
    _silu_gap_flatten_kernel[grid](
        in_0,
        out,
        C,
        HW,
        W,
        in_0.stride(0),
        in_0.stride(1),
        in_0.stride(2),
        in_0.stride(3),
        out.stride(0),
        out.stride(1),
    )
    return out


def replacement_func():
    return fused_silu_adaptive_avg_pool2d_flatten