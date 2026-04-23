import torch
import triton
import triton.language as tl


def pattern(conv_out, gamma, residual):
    tmp_8 = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    tmp_9 = tmp_8 * gamma
    tmp_10 = residual + tmp_9
    return tmp_10


def replacement_args(conv_out, gamma, residual):
    return (conv_out, gamma, residual)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 1024}, num_warps=8, num_stages=2),
    ],
    key=["HW"],
)
@triton.jit
def fused_post_conv_scale_residual_kernel(
    conv_ptr,
    gamma_ptr,
    residual_ptr,
    out_ptr,
    HW,
    C,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_nc = tl.program_id(1)
    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < HW
    c = pid_nc % C
    base = pid_nc * HW + offs
    conv = tl.load(conv_ptr + base, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + base, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + c).to(tl.float32)
    out = residual + conv * gamma
    tl.store(out_ptr + base, out, mask=mask)


@torch.fx.wrap
def fused_post_conv_scale_residual(conv_out, gamma, residual):
    out = torch.empty_like(conv_out)
    hw = conv_out.shape[2] * conv_out.shape[3]
    c = conv_out.shape[1]
    grid = lambda meta: (triton.cdiv(hw, meta["BLOCK_HW"]), conv_out.shape[0] * c)
    fused_post_conv_scale_residual_kernel[grid](
        conv_out,
        gamma,
        residual,
        out,
        hw,
        c,
    )
    return out


def replacement_func():
    return fused_post_conv_scale_residual