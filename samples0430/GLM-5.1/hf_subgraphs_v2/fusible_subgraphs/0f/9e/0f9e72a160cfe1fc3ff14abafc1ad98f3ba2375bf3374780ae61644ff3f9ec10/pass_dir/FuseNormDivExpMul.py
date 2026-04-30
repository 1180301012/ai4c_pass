import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_1 = in_1.norm(p = 2, dim = -1, keepdim = True)
    tmp_2 = in_1 / tmp_1
    tmp_3 = in_2.norm(p = 2, dim = -1, keepdim = True)
    tmp_4 = in_2 / tmp_3
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * tmp_4
    return (tmp_6, tmp_4, tmp_2)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_norm_exp_mul_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_6_ptr,
    out_4_ptr,
    out_2_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load scale (scalar) and compute exp
    scale = tl.load(in_0_ptr).to(tl.float32)
    exp_scale = tl.math.exp(scale)

    # Load in_1 and compute L2 norm along last dimension
    in_1_vals = tl.load(in_1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_sq_1 = tl.sum(in_1_vals * in_1_vals)
    norm_1 = tl.math.sqrt(sum_sq_1)
    norm_in_1 = in_1_vals / norm_1

    # Load in_2 and compute L2 norm along last dimension
    in_2_vals = tl.load(in_2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_sq_2 = tl.sum(in_2_vals * in_2_vals)
    norm_2 = tl.math.sqrt(sum_sq_2)
    norm_in_2 = in_2_vals / norm_2

    # Compute scaled output: exp(in_0) * normalized_in_2
    scaled = exp_scale * norm_in_2

    # Store results
    tl.store(out_2_ptr + offsets, norm_in_1, mask=mask)
    tl.store(out_4_ptr + offsets, norm_in_2, mask=mask)
    tl.store(out_6_ptr + offsets, scaled, mask=mask)


@torch.fx.wrap
def fused_norm_exp_mul(in_0, in_1, in_2):
    # Allocate output tensors with same shape and dtype as expected outputs
    out_2 = torch.empty_like(in_1)   # normalized in_1, shape [1, 512]
    out_4 = torch.empty_like(in_2)   # normalized in_2, shape [1, 1, 512]
    out_6 = torch.empty_like(in_2)   # exp(in_0) * normalized_in_2, shape [1, 1, 512]

    n = in_1.shape[-1]  # 512
    BLOCK_SIZE = triton.next_power_of_2(n)

    fused_norm_exp_mul_kernel[(1,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_6_ptr=out_6,
        out_4_ptr=out_4,
        out_2_ptr=out_2,
        n_elements=n,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_6, out_4, out_2


def replacement_func():
    return fused_norm_exp_mul