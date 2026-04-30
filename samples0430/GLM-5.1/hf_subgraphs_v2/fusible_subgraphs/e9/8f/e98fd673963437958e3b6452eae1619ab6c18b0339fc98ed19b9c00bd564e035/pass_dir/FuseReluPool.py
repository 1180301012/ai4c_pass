import torch
import triton
import triton.language as tl


def pattern(tmp_input):
    tmp_5 = torch.nn.functional.relu(tmp_input, inplace=True)
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return (tmp_7,)


def replacement_args(tmp_input):
    return (tmp_input,)


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
def fused_relu_pool_kernel(
    in_ptr, out_ptr,
    C, H, W,
    stride_in_c, stride_in_h, stride_in_w,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """Fused kernel for relu + adaptive_avg_pool2d(output_size=1) + flatten."""
    pid = tl.program_id(0)

    c_offsets = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    HW = H * W

    # Accumulate sum for pooling in float32
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    # Iterate over spatial positions in blocks
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW

        h_idx = hw_offsets // W
        w_idx = hw_offsets % W

        in_off = c_offsets[:, None] * stride_in_c + h_idx[None, :] * stride_in_h + w_idx[None, :] * stride_in_w

        mask_2d = c_mask[:, None] & hw_mask[None, :]

        # Load input in float32
        in_vals = tl.load(in_ptr + in_off, mask=mask_2d, other=0.0).to(tl.float32)

        # Apply ReLU
        result = tl.maximum(in_vals, 0.0)

        # Accumulate sum over spatial dimensions for each channel
        acc += tl.sum(result, axis=1)

    # Divide by HW to get average
    avg = acc / HW

    # Store result - output shape [1, C] after flatten
    tl.store(out_ptr + c_offsets, avg, mask=c_mask)


@torch.fx.wrap
def fused_relu_pool(in_tensor):
    N, C, H, W = in_tensor.shape
    HW = H * W

    out = torch.empty(N, C, dtype=in_tensor.dtype, device=in_tensor.device)

    stride_in_c = in_tensor.stride(1)
    stride_in_h = in_tensor.stride(2)
    stride_in_w = in_tensor.stride(3)

    BLOCK_C = 4
    BLOCK_HW = 64

    grid = (triton.cdiv(C, BLOCK_C),)

    fused_relu_pool_kernel[grid](
        in_tensor, out,
        C, H, W,
        stride_in_c, stride_in_h, stride_in_w,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
    )

    return (out,)


def replacement_func():
    return fused_relu_pool