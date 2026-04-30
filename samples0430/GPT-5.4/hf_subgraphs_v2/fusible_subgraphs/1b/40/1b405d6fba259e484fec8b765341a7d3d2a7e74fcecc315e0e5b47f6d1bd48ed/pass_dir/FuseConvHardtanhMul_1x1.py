import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv2d
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_P": 64, "BLOCK_CO": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_P": 128, "BLOCK_CO": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_P": 128, "BLOCK_CO": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_P": 256, "BLOCK_CO": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_P": 256, "BLOCK_CO": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["P", "CO", "CI"],
)
@triton.jit
def _fused_conv1x1_hardtanh_mul_kernel(
    bias_ptr,
    weight_ptr,
    x_ptr,
    gate_ptr,
    out_ptr,
    P,
    CO,
    CI,
    W,
    HW,
    bias_s0,
    w_s0,
    w_s1,
    x_s0,
    x_s1,
    x_s2,
    x_s3,
    gate_s0,
    gate_s1,
    gate_s2,
    gate_s3,
    out_s0,
    out_s1,
    out_s2,
    out_s3,
    BLOCK_P: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_co = tl.program_id(1)

    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_k = tl.arange(0, BLOCK_K)

    mask_p = offs_p < P
    mask_co = offs_co < CO
    mask_k = offs_k < CI

    n_idx = offs_p // HW
    hw_idx = offs_p % HW
    h_idx = hw_idx // W
    w_idx = hw_idx % W

    x_base = n_idx * x_s0 + h_idx * x_s2 + w_idx * x_s3
    gate_base = n_idx * gate_s0 + h_idx * gate_s2 + w_idx * gate_s3
    out_base = n_idx * out_s0 + h_idx * out_s2 + w_idx * out_s3

    a_ptrs = x_ptr + x_base[:, None] + offs_k[None, :] * x_s1
    b_ptrs = weight_ptr + offs_k[:, None] * w_s1 + offs_co[None, :] * w_s0

    a = tl.load(a_ptrs, mask=mask_p[:, None] & mask_k[None, :], other=0.0)
    b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_co[None, :], other=0.0)

    acc = tl.dot(a, b)

    bias = tl.load(bias_ptr + offs_co * bias_s0, mask=mask_co, other=0.0)
    acc = acc + bias[None, :]

    gate_ptrs = gate_ptr + gate_base[:, None] + offs_co[None, :] * gate_s1
    gate = tl.load(gate_ptrs, mask=mask_p[:, None] & mask_co[None, :], other=0.0)
    gate = tl.maximum(tl.minimum(gate, 6.0), 0.0)

    out = acc * gate

    out_ptrs = out_ptr + out_base[:, None] + offs_co[None, :] * out_s1
    tl.store(out_ptrs, out, mask=mask_p[:, None] & mask_co[None, :])


@torch.fx.wrap
def fused_conv_hardtanh_mul_1x1(bias, weight, x, gate):
    out = torch.empty_like(gate)

    N = x.shape[0]
    CI = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    CO = gate.shape[1]
    HW = H * W
    P = N * HW

    grid = lambda meta: (
        triton.cdiv(P, meta["BLOCK_P"]),
        triton.cdiv(CO, meta["BLOCK_CO"]),
    )

    _fused_conv1x1_hardtanh_mul_kernel[grid](
        bias,
        weight,
        x,
        gate,
        out,
        P,
        CO,
        CI,
        W,
        HW,
        bias.stride(0),
        weight.stride(0),
        weight.stride(1),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        gate.stride(0),
        gate.stride(1),
        gate.stride(2),
        gate.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return out


def replacement_func():
    return fused_conv_hardtanh_mul_1x1