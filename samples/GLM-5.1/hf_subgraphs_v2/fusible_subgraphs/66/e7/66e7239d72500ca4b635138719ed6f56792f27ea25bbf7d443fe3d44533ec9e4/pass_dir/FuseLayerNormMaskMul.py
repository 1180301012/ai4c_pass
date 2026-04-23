import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    tmp_4 = torch.nn.functional.layer_norm(in_3, (768,), in_2, in_1, 1e-12)
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.expand_as(tmp_4)
    tmp_7 = tmp_6.float()
    tmp_8 = tmp_4 * tmp_7
    return tmp_8

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def layernorm_mask_mul_kernel(
    x_ptr,
    mask_ptr,
    weight_ptr,
    bias_ptr,
    product_ptr,
    n_rows,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + row_idx * N + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * rstd

    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    ln_result = x_norm * weight + bias

    mask_val = tl.load(mask_ptr + row_idx).to(tl.float32)
    product = ln_result * mask_val

    if IS_BF16:
        product_store = product.to(tl.bfloat16)
    else:
        product_store = product.to(tl.float16)

    tl.store(product_ptr + row_idx * N + offsets, product_store, mask=mask)

@torch.fx.wrap
def fused_layernorm_mask_mul(in_0, in_1, in_2, in_3):
    N = 768
    eps = 1e-12
    shape = in_3.shape
    rows = shape[0] * shape[1]
    is_bf16 = in_3.dtype == torch.bfloat16

    product = torch.empty(shape, dtype=in_3.dtype, device=in_3.device)

    x_input = in_3.contiguous().view(rows, N)
    mask_flat = in_0.contiguous().view(rows)
    weight_input = in_2.contiguous()
    bias_input = in_1.contiguous()
    product_flat = product.view(rows, N)

    BLOCK_SIZE = triton.next_power_of_2(N)

    grid = (rows,)

    layernorm_mask_mul_kernel[grid](
        x_input, mask_flat, weight_input, bias_input,
        product_flat,
        n_rows=rows,
        N=N, eps=eps, BLOCK_SIZE=BLOCK_SIZE, IS_BF16=is_bf16,
    )

    return product

def replacement_func():
    return fused_layernorm_mask_mul