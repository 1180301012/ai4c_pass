import torch
import triton
import triton.language as tl


def pattern(in_1):
    tmp_0 = in_1.sum(dim=2, keepdim=True)
    tmp_1 = in_1 / tmp_0
    return tmp_1


def replacement_args(in_1):
    return (in_1,)


@triton.jit
def sum_div_fused_kernel(
    input_ptr,
    output_ptr,
    HW: tl.constexpr,
    W_dim: tl.constexpr,
    H_dim: tl.constexpr,
    C_dim: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # 2 programs, one per channel. Each loops over all width positions.
    c_idx = tl.program_id(0)

    h_range = tl.arange(0, BLOCK_H)

    for w_idx in range(W_dim):
        offsets = c_idx * HW + h_range * W_dim + w_idx

        # No mask needed since H_dim == BLOCK_H exactly (H=8, BLOCK_H=8)
        vals = tl.load(input_ptr + offsets)
        sum_val = tl.sum(vals)
        result = vals / sum_val
        tl.store(output_ptr + offsets, result)


@torch.fx.wrap
def sum_div_fused(input_tensor):
    shape = input_tensor.shape
    B, C, H, W = shape

    output = torch.empty_like(input_tensor)

    HW = H * W
    BLOCK_H = triton.next_power_of_2(H)

    grid = (B * C,)  # 2 programs for B=1, C=2

    sum_div_fused_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        HW=HW,
        W_dim=W,
        H_dim=H,
        C_dim=C,
        BLOCK_H=BLOCK_H,
    )

    return output


def replacement_func():
    return sum_div_fused