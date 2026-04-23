import torch
import triton
import triton.language as tl


# Match:
#   conv2d(in_3, in_1, in_0, (1,1), (0,0), (1,1), 1)
#   -> add scalar
#   -> div scalar
#   -> clamp_(0, 1)
#   -> multiply with broadcast tensor in_2
# This parameterized pattern matches both:
#   (x + 1.0) / 2.0
# and
#   (x + 3.0) / 6.0

def pattern(in_0, in_1, in_2, in_3, add_const, div_const):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d + add_const
    tmp_4 = tmp_3 / div_const
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, add_const, div_const):
    return (in_0, in_1, in_2, in_3, add_const, div_const)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 32, "BLOCK_HW": 64}, num_warps=2),
        triton.Config({"BLOCK_K": 32, "BLOCK_HW": 128}, num_warps=4),
        triton.Config({"BLOCK_K": 64, "BLOCK_HW": 128}, num_warps=4),
        triton.Config({"BLOCK_K": 64, "BLOCK_HW": 256}, num_warps=4),
        triton.Config({"BLOCK_K": 64, "BLOCK_HW": 512}, num_warps=8),
    ],
    key=["CIN", "HW"],
)
@triton.jit
def _fused_se_gate_scale_kernel(
    x2_ptr,
    x3_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    add_const,
    div_const,
    HW,
    CIN,
    COUT,
    BLOCK_K: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_n = tl.program_id(1)

    acc = tl.zeros((), dtype=tl.float32)

    for k_start in range(0, CIN, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < CIN

        x = tl.load(
            x3_ptr + pid_n * CIN + offs_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            w_ptr + pid_c * CIN + offs_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(x * w, axis=0)

    bias = tl.load(b_ptr + pid_c).to(tl.float32)
    gate = (acc + bias + add_const) / div_const
    gate = tl.maximum(0.0, tl.minimum(1.0, gate))

    base = (pid_n * COUT + pid_c) * HW
    for hw_start in range(0, HW, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        mask_hw = offs_hw < HW
        x2 = tl.load(x2_ptr + base + offs_hw, mask=mask_hw, other=0.0)
        out = x2 * gate
        tl.store(out_ptr + base + offs_hw, out, mask=mask_hw)


@torch.fx.wrap
def fused_se_pointwise_conv_hsigmoid_mul(in_0, in_1, in_2, in_3, add_const, div_const):
    n = in_2.shape[0]
    cout = in_2.shape[1]
    hw = in_2.shape[2] * in_2.shape[3]
    cin = in_3.shape[1]

    out = torch.empty_like(in_2)

    _fused_se_gate_scale_kernel[(cout, n)](
        x2_ptr=in_2,
        x3_ptr=in_3,
        w_ptr=in_1,
        b_ptr=in_0,
        out_ptr=out,
        add_const=add_const,
        div_const=div_const,
        HW=hw,
        CIN=cin,
        COUT=cout,
    )

    return out


def replacement_func():
    return fused_se_pointwise_conv_hsigmoid_mul