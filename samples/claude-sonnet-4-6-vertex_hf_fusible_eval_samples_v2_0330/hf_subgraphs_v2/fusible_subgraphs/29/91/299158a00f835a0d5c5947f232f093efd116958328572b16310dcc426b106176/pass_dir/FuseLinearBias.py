import torch
import triton
import triton.language as tl


def pattern(x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias)


def replacement_args(x, weight, bias):
    return (x, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_F': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_F': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_F': 1024}, num_warps=8, num_stages=2),
    ],
    key=['N', 'F'],
)
@triton.jit
def linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    B, N, F,
    stride_xb, stride_xf,
    stride_wn, stride_wf,
    stride_ob, stride_on,
    BLOCK_F: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    f_offsets = tl.arange(0, BLOCK_F)
    mask_f = f_offsets < F

    x = tl.load(
        x_ptr + pid_b * stride_xb + f_offsets * stride_xf,
        mask=mask_f, other=0.0
    ).to(tl.float32)

    w = tl.load(
        w_ptr + pid_n * stride_wn + f_offsets * stride_wf,
        mask=mask_f, other=0.0
    ).to(tl.float32)

    dot = tl.sum(x * w)
    b = tl.load(b_ptr + pid_n).to(tl.float32)

    out = dot + b

    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + pid_b * stride_ob + pid_n * stride_on, out.to(out_dtype))


@torch.fx.wrap
def triton_linear(x, weight, bias):
    # x: [B, F], weight: [N, F], bias: [N]
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1]) if x.ndim > 2 else x

    B = x_2d.shape[0]
    N = weight.shape[0]
    F = weight.shape[1]

    output = torch.empty(B, N, dtype=x.dtype, device=x.device)

    grid = (B, N)

    linear_kernel[grid](
        x_2d, weight, bias, output,
        B, N, F,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
    )

    if x.ndim > 2:
        output = output.view(*orig_shape[:-1], N)

    return output


def replacement_func():
    return triton_linear