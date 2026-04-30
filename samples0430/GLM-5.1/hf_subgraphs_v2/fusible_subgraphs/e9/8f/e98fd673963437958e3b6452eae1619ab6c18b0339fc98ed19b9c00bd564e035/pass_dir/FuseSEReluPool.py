import torch
import triton
import triton.language as tl


def pattern(in_1, in_2):
    tmp_0 = in_2.sigmoid()
    tmp_1 = tmp_0.view(1, -1, 1, 1)
    tmp_2 = tmp_1.expand_as(in_1)
    tmp_3 = in_1 * tmp_2
    return (tmp_3,)


def replacement_args(in_1, in_2):
    return (in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 1, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 2, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 4, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 8, 'BLOCK_HW': 64}, num_warps=8),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 64}, num_warps=8),
    ],
    key=['C', 'H', 'W'],
)
@triton.jit
def fused_se_kernel(
    in_1_ptr, in_2_ptr, out_ptr,
    C, H, W,
    stride_in1_c, stride_in1_h, stride_in1_w,
    stride_in2_cc,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    c_offsets = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    # Load in_2 values and compute sigmoid
    in_2_vals = tl.load(in_2_ptr + c_offsets * stride_in2_cc, mask=c_mask, other=0.0)
    sig_vals = tl.sigmoid(in_2_vals.to(tl.float32))

    HW = H * W

    # Iterate over spatial positions in blocks
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW

        h_idx = hw_offsets // W
        w_idx = hw_offsets % W

        in_1_off = c_offsets[:, None] * stride_in1_c + h_idx[None, :] * stride_in1_h + w_idx[None, :] * stride_in1_w

        mask_2d = c_mask[:, None] & hw_mask[None, :]

        in_1_vals = tl.load(in_1_ptr + in_1_off, mask=mask_2d, other=0.0).to(tl.float32)

        # Compute: in_1 * sigmoid(in_2)
        result = in_1_vals * sig_vals[:, None]

        # Store result with same shape as in_1
        tl.store(out_ptr + in_1_off, result, mask=mask_2d)


@torch.fx.wrap
def fused_se(in_1, in_2):
    N, C, H, W = in_1.shape

    out = torch.empty_like(in_1)

    stride_in1_c = in_1.stride(1)
    stride_in1_h = in_1.stride(2)
    stride_in1_w = in_1.stride(3)
    stride_in2_cc = in_2.stride(2)

    grid = lambda meta: (triton.cdiv(C, meta['BLOCK_C']),)

    fused_se_kernel[grid](
        in_1, in_2, out,
        C, H, W,
        stride_in1_c, stride_in1_h, stride_in1_w,
        stride_in2_cc,
    )

    return out


def replacement_func():
    return fused_se