import torch
import triton
import triton.language as tl


# Match:
#   conv2d(in_3, in_1, in_0, (1,1), (0,0), (1,1), 1)
#   -> add scalar
#   -> div scalar
#   -> clamp_(0, 1)
#   -> multiply with broadcast tensor in_2
# This parameterized pattern is intended to match both:
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
        triton.Config({"BLOCK_C": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_C": 128, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_C": 128, "BLOCK_K": 64}, num_warps=8),
    ],
    key=["COUT", "CIN"],
)
@triton.jit

def _se_gate_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    gate_ptr,
    add_const,
    div_const,
    CIN,
    COUT,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < COUT

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for k_start in range(0, CIN, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < CIN

        x = tl.load(
            x_ptr + pid_n * CIN + offs_k,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)

        w = tl.load(
            w_ptr + offs_c[:, None] * CIN + offs_k[None, :],
            mask=mask_c[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(w * x[None, :], axis=1)

    bias = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    val = acc + bias
    val = (val + add_const) / div_const
    val = tl.maximum(0.0, tl.minimum(1.0, val))

    tl.store(gate_ptr + pid_n * COUT + offs_c, val, mask=mask_c)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 128}, num_warps=4),
        triton.Config({"BLOCK_HW": 256}, num_warps=4),
        triton.Config({"BLOCK_HW": 512}, num_warps=8),
    ],
    key=["HW"],
)
@triton.jit

def _broadcast_mul_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW

    gate = tl.load(gate_ptr + pid_nc)
    base = pid_nc * HW + offs_hw
    x = tl.load(x_ptr + base, mask=mask_hw, other=0.0)
    out = x * gate
    tl.store(out_ptr + base, out, mask=mask_hw)


@torch.fx.wrap
def fused_se_pointwise_conv_hsigmoid_mul(in_0, in_1, in_2, in_3, add_const, div_const):
    # Shapes:
    # in_0: [COUT]
    # in_1: [COUT, CIN, 1, 1]
    # in_2: [N, COUT, H, W]
    # in_3: [N, CIN, 1, 1]
    n = in_2.shape[0]
    cout = in_2.shape[1]
    h = in_2.shape[2]
    w = in_2.shape[3]
    hw = h * w
    cin = in_3.shape[1]

    gate = torch.empty((n, cout), device=in_2.device, dtype=in_2.dtype)
    out = torch.empty_like(in_2)

    _se_gate_kernel[
        lambda meta: (n, triton.cdiv(cout, meta["BLOCK_C"]))
    ](
        x_ptr=in_3,
        w_ptr=in_1,
        b_ptr=in_0,
        gate_ptr=gate,
        add_const=add_const,
        div_const=div_const,
        CIN=cin,
        COUT=cout,
    )

    _broadcast_mul_kernel[
        lambda meta: (n * cout, triton.cdiv(hw, meta["BLOCK_HW"]))
    ](
        x_ptr=in_2,
        gate_ptr=gate,
        out_ptr=out,
        HW=hw,
    )

    return out


def replacement_func():
    return fused_se_pointwise_conv_hsigmoid_mul