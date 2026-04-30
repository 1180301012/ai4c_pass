import torch
import triton
import triton.language as tl


# Matches:
#   tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
#   tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
# We intentionally stop at tmp_1 because the following view/flatten are metadata-only.
def pattern(in_0):
    tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 16, "BLOCK_HW": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 32, "BLOCK_HW": 32}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_C": 32, "BLOCK_HW": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_C": 32, "BLOCK_HW": 128}, num_warps=8, num_stages=1),
    ],
    key=["c", "hw"],
)
@triton.jit
def _hardtanh_gap_contiguous_blocked_kernel(
    x_ptr,
    out_ptr,
    c,
    hw,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    MAX_HW: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < c
    hw_range = tl.arange(0, BLOCK_HW)

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    base_nc = (pid_n * c + c_offsets)[:, None] * hw

    for hw_start in tl.static_range(0, MAX_HW, BLOCK_HW):
        offs_hw = hw_start + hw_range
        mask = mask_c[:, None] & (offs_hw[None, :] < hw)
        ptrs = x_ptr + base_nc + offs_hw[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        vals = tl.minimum(tl.maximum(vals, 0.0), 6.0)
        acc += tl.sum(vals, axis=1)

    out_ptrs = out_ptr + pid_n * c + c_offsets
    tl.store(out_ptrs, acc / hw, mask=mask_c)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 8, "BLOCK_HW": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 16, "BLOCK_HW": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_C": 32, "BLOCK_HW": 32}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_C": 32, "BLOCK_HW": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_C": 32, "BLOCK_HW": 128}, num_warps=8, num_stages=1),
    ],
    key=["c", "h", "w"],
)
@triton.jit
def _hardtanh_gap_strided_blocked_kernel(
    x_ptr,
    out_ptr,
    c,
    h,
    w,
    x_s0,
    x_s1,
    x_s2,
    x_s3,
    out_s0,
    out_s1,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    MAX_HW: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < c
    hw_range = tl.arange(0, BLOCK_HW)
    hw = h * w

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    base_ptrs = x_ptr + pid_n * x_s0 + c_offsets[:, None] * x_s1

    for hw_start in tl.static_range(0, MAX_HW, BLOCK_HW):
        offs_hw = hw_start + hw_range
        mask = mask_c[:, None] & (offs_hw[None, :] < hw)
        h_idx = offs_hw // w
        w_idx = offs_hw % w
        ptrs = base_ptrs + h_idx[None, :] * x_s2 + w_idx[None, :] * x_s3
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        vals = tl.minimum(tl.maximum(vals, 0.0), 6.0)
        acc += tl.sum(vals, axis=1)

    out_ptrs = out_ptr + pid_n * out_s0 + c_offsets * out_s1
    tl.store(out_ptrs, acc / hw, mask=mask_c)


@torch.fx.wrap
def fused_hardtanh_gap_1x1(in_0):
    n, c, h, w = in_0.shape
    out = torch.empty((n, c, 1, 1), device=in_0.device, dtype=in_0.dtype)

    hw = h * w
    if hw <= 16:
        max_hw = 16
    elif hw <= 32:
        max_hw = 32
    elif hw <= 64:
        max_hw = 64
    elif hw <= 128:
        max_hw = 128
    elif hw <= 256:
        max_hw = 256
    else:
        max_hw = 512

    grid = lambda META: (triton.cdiv(c, META["BLOCK_C"]), n)

    if in_0.is_contiguous():
        _hardtanh_gap_contiguous_blocked_kernel[grid](
            in_0,
            out,
            c,
            hw,
            MAX_HW=max_hw,
        )
    else:
        _hardtanh_gap_strided_blocked_kernel[grid](
            in_0,
            out,
            c,
            h,
            w,
            in_0.stride(0),
            in_0.stride(1),
            in_0.stride(2),
            in_0.stride(3),
            out.stride(0),
            out.stride(1),
            MAX_HW=max_hw,
        )

    return out


def replacement_func():
    return fused_hardtanh_gap_1x1