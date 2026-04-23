import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(8, -1)
    tmp_2 = tmp_1.view(8, -1, 1, 1)
    tmp_3 = tmp_2.view(8, 2, -1, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1, "b8")


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 64}, num_warps=2),
        triton.Config({"BLOCK_HW": 128}, num_warps=4),
        triton.Config({"BLOCK_HW": 256}, num_warps=4),
        triton.Config({"BLOCK_HW": 512}, num_warps=8),
    ],
    key=["HW"],
)
@triton.jit
def _fused_softmax_weighted_sum_kernel(
    in0_ptr,
    gate_ptr,
    out_ptr,
    B,
    C,
    H,
    W,
    HW,
    in0_s0,
    in0_s1,
    in0_s2,
    in0_s3,
    in0_s4,
    gate_s0,
    gate_s1,
    gate_s2,
    gate_s3,
    out_s0,
    out_s1,
    out_s2,
    out_s3,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs_hw < HW

    h_idx = offs_hw // W
    w_idx = offs_hw - h_idx * W

    gate_base = pid_b * gate_s0 + pid_c * gate_s3
    l0 = tl.load(gate_ptr + gate_base + 0 * gate_s1 + 0 * gate_s2).to(tl.float32)
    l1 = tl.load(gate_ptr + gate_base + 1 * gate_s1 + 0 * gate_s2).to(tl.float32)

    m = tl.maximum(l0, l1)
    e0 = tl.exp(l0 - m)
    e1 = tl.exp(l1 - m)
    denom = e0 + e1
    w0 = e0 / denom
    w1 = e1 / denom

    in0_base = pid_b * in0_s0 + pid_c * in0_s2 + h_idx * in0_s3 + w_idx * in0_s4
    x0 = tl.load(in0_ptr + in0_base + 0 * in0_s1, mask=mask, other=0).to(tl.float32)
    x1 = tl.load(in0_ptr + in0_base + 1 * in0_s1, mask=mask, other=0).to(tl.float32)
    y = x0 * w0 + x1 * w1

    out_base = pid_b * out_s0 + pid_c * out_s1 + h_idx * out_s2 + w_idx * out_s3
    tl.store(out_ptr + out_base, y, mask=mask)


@torch.fx.wrap
def fused_softmax_weighted_sum_dispatch(in_0, in_1, route):
    if route == "b1":
        pass
    elif route == "b2":
        pass
    elif route == "b8":
        pass

    B = in_0.shape[0]
    C = in_0.shape[2]
    H = in_0.shape[3]
    W = in_0.shape[4]
    HW = H * W

    out = torch.empty((B, C, H, W), device=in_0.device, dtype=in_0.dtype)

    grid = lambda META: (triton.cdiv(HW, META["BLOCK_HW"]), C, B)

    _fused_softmax_weighted_sum_kernel[grid](
        in_0,
        in_1,
        out,
        B,
        C,
        H,
        W,
        HW,
        in_0.stride(0),
        in_0.stride(1),
        in_0.stride(2),
        in_0.stride(3),
        in_0.stride(4),
        in_1.stride(0),
        in_1.stride(1),
        in_1.stride(2),
        in_1.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return out


def replacement_func():
    return fused_softmax_weighted_sum_dispatch