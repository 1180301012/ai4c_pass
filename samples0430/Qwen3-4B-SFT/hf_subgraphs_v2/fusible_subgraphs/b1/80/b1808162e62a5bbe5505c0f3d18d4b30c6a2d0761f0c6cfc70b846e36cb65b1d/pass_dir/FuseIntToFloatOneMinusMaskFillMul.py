import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = 1.0 - tmp_0
    tmp_2 = tmp_1.bool()
    tmp_3 = tmp_1.masked_fill(tmp_2, -3.4028234663852886e+38)
    tmp_4 = tmp_3 * tmp_1
    return (tmp_4,)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_int_to_float_one_minus_mask_fill_mul_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load int64 input
    x = tl.load(in_ptr + offsets, mask=mask)

    # Cast to float32 and compute 1.0 - x
    x_f32 = x.to(tl.float32)
    one_minus_x = 1.0 - x_f32

    # masked_fill + multiply: where (1-x) != 0, result = -3.4e38*(1-x), else 0
    filled = tl.where(one_minus_x != 0.0,
                      one_minus_x * (-3.4028234663852886e+38),
                      one_minus_x * 0.0)

    tl.store(out_ptr + offsets, filled, mask=mask)


@torch.fx.wrap
def fused_int_to_float_one_minus_mask_fill_mul(in_0):
    shape = in_0.shape
    n_elements = in_0.numel()

    # Output is float32 (same shape as the downstream .to(float32))
    out = torch.empty(shape, dtype=torch.float32, device=in_0.device)

    BLOCK_SIZE = 512
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    fused_int_to_float_one_minus_mask_fill_mul_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=16,
    )

    return out


def replacement_func():
    return fused_int_to_float_one_minus_mask_fill_mul