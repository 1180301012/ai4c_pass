import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_layernorm_768_rev_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr, out_ptr,
    N, eps,
    stride_row,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    # Load inputs, upcast to fp32 immediately
    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    z = x + y

    # Mean: masked positions have z=0
    mean = tl.sum(z, axis=0) / N

    # Variance: zero out masked positions
    diff = tl.where(mask, z - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Affine parameters
    w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    result = diff * inv_std * w + b

    # Store — cast back to the input element type
    tl.store(out_ptr + row * stride_row + offs, result.to(x_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def fused_add_layernorm_768_rev(in_0, in_1, in_2, in_3):
    # in_0: bias  [768]
    # in_1: weight [768]
    # in_2, in_3: inputs [*, 768]  (add order: in_3 + in_2, commutative)
    N = 768
    BLOCK_N = 1024  # next power-of-2 >= 768
    num_rows = in_3.numel() // N
    out = torch.empty_like(in_3)

    fused_add_layernorm_768_rev_kernel[(num_rows,)](
        in_3, in_2, in_1, in_0, out,
        N, 1e-05,
        N,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )

    return out


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-05)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_add_layernorm_768_rev