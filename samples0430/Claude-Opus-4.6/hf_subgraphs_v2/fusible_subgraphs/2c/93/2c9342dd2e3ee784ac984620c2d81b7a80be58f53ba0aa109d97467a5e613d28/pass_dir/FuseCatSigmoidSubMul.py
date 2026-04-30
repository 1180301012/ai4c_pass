import torch
import triton
import triton.language as tl


def pattern(bias, weight, input_tensor, batch_size, view_4, view_5):
    conv = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    viewed = conv.view(batch_size, 1, -1)
    cat = torch.cat([view_4, view_5, viewed], 2)
    sig = cat.sigmoid()
    sub = sig - 0.25
    mul = sub * 3.141592653589793
    return mul


def replacement_args(bias, weight, input_tensor, batch_size, view_4, view_5):
    return (bias, weight, input_tensor, view_4, view_5)


@triton.jit
def full_fused_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    in_3_ptr,
    in_4_ptr,
    out_ptr,
    size_in_3: tl.constexpr,
    size_in_4: tl.constexpr,
    total_per_batch: tl.constexpr,
    in_channels: tl.constexpr,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: (blocks_per_batch, B)
    batch_idx = tl.program_id(1)
    block_in_batch = tl.program_id(0)
    pos_idx = block_in_batch * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pos_idx < total_per_batch

    block_start = block_in_batch * BLOCK_SIZE
    conv_start: tl.constexpr = size_in_3 + size_in_4

    # FAST PATH: entire block is in in_3 region (majority of blocks ~67%)
    if block_start + BLOCK_SIZE <= size_in_3:
        offset = batch_idx * size_in_3 + pos_idx
        val = tl.load(in_3_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    else:
        # Determine which source region each element belongs to
        is_in_3 = pos_idx < size_in_3
        is_in_4 = (pos_idx >= size_in_3) & (pos_idx < conv_start)
        is_conv = pos_idx >= conv_start

        # Load from in_3
        in_3_offset = batch_idx * size_in_3 + pos_idx
        val_3 = tl.load(in_3_ptr + in_3_offset, mask=mask & is_in_3, other=0.0)

        # Load from in_4
        in_4_offset = batch_idx * size_in_4 + (pos_idx - size_in_3)
        val_4 = tl.load(in_4_ptr + in_4_offset, mask=mask & is_in_4, other=0.0)

        # Compute conv only for blocks that overlap the conv region
        conv_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        if block_start + BLOCK_SIZE > conv_start:
            spatial_idx = pos_idx - conv_start
            for c in range(in_channels):
                x_offset = batch_idx * in_channels * spatial_size + c * spatial_size + spatial_idx
                x = tl.load(input_ptr + x_offset, mask=mask & is_conv, other=0.0)
                w = tl.load(weight_ptr + c)
                conv_val += x.to(tl.float32) * w.to(tl.float32)
            bias_val = tl.load(bias_ptr)
            conv_val = conv_val + bias_val.to(tl.float32)

        # Select the correct value
        val = tl.where(is_in_3, val_3.to(tl.float32), tl.where(is_in_4, val_4.to(tl.float32), conv_val))

    # Apply sigmoid + subtract + multiply
    val = tl.sigmoid(val)
    val = (val - 0.25) * 3.141592653589793

    # Store result
    out_offset = batch_idx * total_per_batch + pos_idx
    tl.store(out_ptr + out_offset, val, mask=mask)


@torch.fx.wrap
def full_fused_op(bias, weight, input_tensor, view_4, view_5):
    B = input_tensor.shape[0]
    size_in_3 = view_4.shape[2]
    size_in_4 = view_5.shape[2]
    spatial_size = input_tensor.shape[2] * input_tensor.shape[3]
    total_per_batch = size_in_3 + size_in_4 + spatial_size
    in_channels = input_tensor.shape[1]

    out = torch.empty((B, 1, total_per_batch), dtype=view_4.dtype, device=view_4.device)

    BLOCK_SIZE = 1024
    blocks_per_batch = (total_per_batch + BLOCK_SIZE - 1) // BLOCK_SIZE

    full_fused_kernel[(blocks_per_batch, B)](
        bias, weight, input_tensor, view_4, view_5, out,
        size_in_3, size_in_4, total_per_batch,
        in_channels, spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )

    return out


def replacement_func():
    return full_fused_op