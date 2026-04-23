import torch
import triton
import triton.language as tl


def pattern(in_0, gate):
    tmp_4 = gate * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, gate):
    return (in_0, gate)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 256}, num_warps=4),
        triton.Config({"BLOCK_HW": 512}, num_warps=4),
        triton.Config({"BLOCK_HW": 1024}, num_warps=8),
    ],
    key=["HW"],
)
@triton.jit
def _weighted_branch_reduce_contig_kernel(
    in0_ptr,
    gate_ptr,
    out_ptr,
    BC,
    C,
    HW,
    in0_s0,
    in0_s1,
    in0_s2,
    gate_s0,
    gate_s1,
    gate_s2,
    out_s0,
    out_s1,
    BLOCK_HW: tl.constexpr,
):
    pid_bc = tl.program_id(0)

    b = pid_bc // C
    c = pid_bc - b * C

    gate_base = b * gate_s0 + c * gate_s2
    g0 = tl.load(gate_ptr + gate_base + 0 * gate_s1)
    g1 = tl.load(gate_ptr + gate_base + 1 * gate_s1)

    in0_base = b * in0_s0 + c * in0_s2
    out_base = b * out_s0 + c * out_s1

    for block_idx in range(0, tl.cdiv(HW, BLOCK_HW)):
        offs_hw = block_idx * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask = offs_hw < HW
        x0 = tl.load(in0_ptr + in0_base + 0 * in0_s1 + offs_hw, mask=mask, other=0.0)
        x1 = tl.load(in0_ptr + in0_base + 1 * in0_s1 + offs_hw, mask=mask, other=0.0)
        y = x0 * g0 + x1 * g1
        tl.store(out_ptr + out_base + offs_hw, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 256}, num_warps=4),
        triton.Config({"BLOCK_HW": 512}, num_warps=8),
    ],
    key=["HW"],
)
@triton.jit
def _weighted_branch_reduce_generic_kernel(
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
    gate_s4,
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

    gate_base = pid_b * gate_s0 + pid_c * gate_s2
    g0 = tl.load(gate_ptr + gate_base + 0 * gate_s1 + 0 * gate_s3 + 0 * gate_s4)
    g1 = tl.load(gate_ptr + gate_base + 1 * gate_s1 + 0 * gate_s3 + 0 * gate_s4)

    in0_base = pid_b * in0_s0 + pid_c * in0_s2 + h_idx * in0_s3 + w_idx * in0_s4
    x0 = tl.load(in0_ptr + in0_base + 0 * in0_s1, mask=mask, other=0.0)
    x1 = tl.load(in0_ptr + in0_base + 1 * in0_s1, mask=mask, other=0.0)
    y = x0 * g0 + x1 * g1

    out_base = pid_b * out_s0 + pid_c * out_s1 + h_idx * out_s2 + w_idx * out_s3
    tl.store(out_ptr + out_base, y, mask=mask)


@torch.fx.wrap
def weighted_branch_reduce(in_0, gate):
    B = in_0.shape[0]
    C = in_0.shape[2]
    H = in_0.shape[3]
    W = in_0.shape[4]
    HW = H * W
    BC = B * C

    out = torch.empty((B, C, H, W), device=in_0.device, dtype=in_0.dtype)

    if in_0.is_contiguous() and gate.is_contiguous():
        grid = lambda META: (BC,)
        _weighted_branch_reduce_contig_kernel[grid](
            in_0,
            gate,
            out,
            BC,
            C,
            HW,
            in_0.stride(0),
            in_0.stride(1),
            in_0.stride(2),
            gate.stride(0),
            gate.stride(1),
            gate.stride(2),
            out.stride(0),
            out.stride(1),
        )
    else:
        grid = lambda META: (triton.cdiv(HW, META["BLOCK_HW"]), C, B)
        _weighted_branch_reduce_generic_kernel[grid](
            in_0,
            gate,
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
            gate.stride(0),
            gate.stride(1),
            gate.stride(2),
            gate.stride(3),
            gate.stride(4),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
        )
    return out


def replacement_func():
    return weighted_branch_reduce