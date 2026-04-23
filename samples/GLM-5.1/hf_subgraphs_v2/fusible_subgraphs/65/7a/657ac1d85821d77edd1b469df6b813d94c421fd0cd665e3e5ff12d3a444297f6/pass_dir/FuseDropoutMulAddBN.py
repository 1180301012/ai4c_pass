import torch
import triton
import triton.language as tl


def pattern(conv_out, gamma, residual):
    scaled = conv_out * gamma
    add_out = residual + scaled
    return add_out


def replacement_args(conv_out, gamma, residual):
    return (conv_out, gamma, residual)


@triton.jit
def fused_mul_add_kernel(
    conv_ptr, gamma_ptr, residual_ptr,
    out_ptr,
    n_elements,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    conv_val = tl.load(conv_ptr + offsets, mask=mask, other=0.0)
    c = (offsets // HW) % C
    gamma_val = tl.load(gamma_ptr + c, mask=mask, other=1.0)
    residual_val = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    out_val = residual_val + conv_val * gamma_val
    tl.store(out_ptr + offsets, out_val, mask=mask)


@torch.fx.wrap
def fused_mul_add(conv_out, gamma, residual):
    N, C, H, W = conv_out.shape
    HW = H * W
    n_elements = N * C * H * W

    out = torch.empty_like(conv_out)

    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    fused_mul_add_kernel[grid](
        conv_ptr=conv_out,
        gamma_ptr=gamma,
        residual_ptr=residual,
        out_ptr=out,
        n_elements=n_elements,
        C=C,
        HW=HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_mul_add