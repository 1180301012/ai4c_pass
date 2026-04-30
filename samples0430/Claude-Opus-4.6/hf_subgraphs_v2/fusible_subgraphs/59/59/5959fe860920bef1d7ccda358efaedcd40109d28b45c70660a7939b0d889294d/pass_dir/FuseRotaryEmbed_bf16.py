import torch
import triton
import triton.language as tl


def pattern(in_1):
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    return tmp_6, tmp_7


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def _rotary_bf16_kernel(
    in_ptr,
    cos_ptr,
    sin_ptr,
    num_rows,
    D,
    in_stride,
    out_stride,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    # Load input and cast to fp32 for trig computation
    x = tl.load(in_ptr + row * in_stride + offs, mask=mask, other=0.0).to(tl.float32)

    # Compute cos and sin in fp32, then cast to bf16
    c = tl.cos(x).to(tl.bfloat16)
    s = tl.sin(x).to(tl.bfloat16)

    # Store first half (positions 0..D-1)
    cos_base = cos_ptr + row * out_stride
    sin_base = sin_ptr + row * out_stride
    tl.store(cos_base + offs, c, mask=mask)
    tl.store(sin_base + offs, s, mask=mask)

    # Store second half (positions D..2D-1) - repeated
    tl.store(cos_base + offs + D, c, mask=mask)
    tl.store(sin_base + offs + D, s, mask=mask)


@torch.fx.wrap
def rotary_bf16_fn(in_1):
    shape = in_1.shape
    B = shape[0]
    S = shape[1]
    D = shape[2]
    D2 = D * 2

    cos_out = torch.empty(B, S, D2, dtype=torch.bfloat16, device=in_1.device)
    sin_out = torch.empty(B, S, D2, dtype=torch.bfloat16, device=in_1.device)

    num_rows = B * S
    BLOCK_D = triton.next_power_of_2(D)

    _rotary_bf16_kernel[(num_rows,)](
        in_1,
        cos_out,
        sin_out,
        num_rows,
        D,
        D,
        D2,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )

    return cos_out, sin_out


def replacement_func():
    return rotary_bf16_fn