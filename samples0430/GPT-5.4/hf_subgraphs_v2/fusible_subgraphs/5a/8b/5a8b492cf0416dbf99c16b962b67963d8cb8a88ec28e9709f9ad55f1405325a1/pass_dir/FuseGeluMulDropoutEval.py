import torch
import triton
import triton.language as tl


# Pattern matching function
# Must mirror model.py exactly and exclude cleanup statements.
def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0 * in_1
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return tmp_2


# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_gelu_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    gelu_x = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    out = gelu_x * y

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_gelu_mul_dropout_eval(in_0, in_1):
    out = torch.empty_like(in_0)
    n_elements = out.numel()

    if n_elements <= 32768:
        block_size = 256
        num_warps = 4
    elif n_elements <= 262144:
        block_size = 512
        num_warps = 4
    else:
        block_size = 1024
        num_warps = 8

    grid = (triton.cdiv(n_elements, block_size),)
    fused_gelu_mul_kernel[grid](
        in_0,
        in_1,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_gelu_mul_dropout_eval