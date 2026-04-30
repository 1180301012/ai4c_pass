import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_relu_maxpool5x5_cat_kernel(
    in_ptr, out_ptr,
    C, H, W,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decode (b, c, h, w) from linear index in [B, C, H, W]
    HW = H * W
    CHW = C * HW

    w_idx = offsets % W
    h_idx = (offsets // W) % H
    c_idx = (offsets // HW) % C
    b_idx = offsets // CHW

    # Load center value
    center_val = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # Compute relu for output's first C channels
    relu_val = tl.maximum(center_val, 0.0)

    # Compute max over 5x5 window for maxpool
    # Key identity: maxpool(relu(x)) = relu(maxpool(x))
    # So we compute maxpool on raw input, then apply relu
    max_val = center_val

    for dh in range(-2, 3):
        for dw in range(-2, 3):
            if dh == 0 and dw == 0:
                continue
            ih = h_idx + dh
            iw = w_idx + dw
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            # Clamp indices for safe memory access
            ih_c = tl.maximum(tl.minimum(ih, H - 1), 0)
            iw_c = tl.maximum(tl.minimum(iw, W - 1), 0)
            load_off = b_idx * CHW + c_idx * HW + ih_c * W + iw_c
            val = tl.load(in_ptr + load_off, mask=mask, other=0.0)
            val = tl.where(valid, val, float('-inf'))
            max_val = tl.maximum(max_val, val)

    # Apply relu to maxpool result: maxpool(relu(x)) = relu(maxpool(x))
    pool_val = tl.maximum(max_val, 0.0)

    # Output layout: [B, 4*C, H, W], contiguous
    out_4CHW = 4 * CHW
    out_base = b_idx * out_4CHW + h_idx * W + w_idx

    # Write relu to first C channels
    out_off_0 = out_base + c_idx * HW
    tl.store(out_ptr + out_off_0, relu_val, mask=mask)

    # Write pool to channels [C, 2C), [2C, 3C), [3C, 4C)
    out_off_1 = out_base + (c_idx + C) * HW
    out_off_2 = out_base + (c_idx + 2 * C) * HW
    out_off_3 = out_base + (c_idx + 3 * C) * HW
    tl.store(out_ptr + out_off_1, pool_val, mask=mask)
    tl.store(out_ptr + out_off_2, pool_val, mask=mask)
    tl.store(out_ptr + out_off_3, pool_val, mask=mask)


@torch.fx.wrap
def fused_relu_maxpool5x5_cat(in_0):
    B, C, H, W = in_0.shape
    n_elements = B * C * H * W

    output = torch.empty(B, 4 * C, H, W, dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_relu_maxpool5x5_cat_kernel[grid](
        in_0, output,
        C, H, W,
        n_elements,
    )

    return output


def replacement_func():
    return fused_relu_maxpool5x5_cat