import torch
import triton
import triton.language as tl


@triton.jit
def fused_avgpool_bn_silu_kernel(
    in_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
    EPSILON: tl.constexpr = 1e-5,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # Unflatten output index to (channel, height, width)
    ch_idx = offsets // (8 * 8)
    rem = offsets % (8 * 8)
    h_idx = rem // 8
    w_idx = rem % 8

    # Calculate input offsets for 2x2 pooling
    base_offset = ch_idx * (16 * 16)
    off0 = base_offset + (2 * h_idx) * 16 + 2 * w_idx
    off1 = base_offset + (2 * h_idx) * 16 + 2 * w_idx + 1
    off2 = base_offset + (2 * h_idx + 1) * 16 + 2 * w_idx
    off3 = base_offset + (2 * h_idx + 1) * 16 + 2 * w_idx + 1

    # Load 4 input elements
    x0 = tl.load(in_ptr + off0, mask=mask, other=0.0)
    x1 = tl.load(in_ptr + off1, mask=mask, other=0.0)
    x2 = tl.load(in_ptr + off2, mask=mask, other=0.0)
    x3 = tl.load(in_ptr + off3, mask=mask, other=0.0)

    # AvgPool (2x2)
    avg = (x0 + x1 + x2 + x3) * 0.25

    # BatchNorm
    mean = tl.load(mean_ptr + ch_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + ch_idx, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + ch_idx, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + ch_idx, mask=mask, other=0.0)
    normalized = (avg - mean) * tl.rsqrt(var + EPSILON)
    bn = normalized * weight + bias

    # SiLU activation
    silu = bn * tl.sigmoid(bn)

    # Store output
    tl.store(out_ptr + offsets, silu, mask=mask)


@torch.fx.wrap
def fused_avgpool_bn_silu(in_4, in_0, in_1, in_2, in_3):
    # Output shape: [1, 512, 8, 8]
    num_elements = 1 * 512 * 8 * 8
    BLOCK_SIZE = 128
    num_blocks = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Create output tensor
    out = torch.empty(1, 512, 8, 8, dtype=in_4.dtype, device=in_4.device)

    # Launch kernel
    fused_avgpool_bn_silu_kernel[(num_blocks,)](
        in_4,
        in_0,
        in_1,
        in_3,
        in_2,
        out,
        num_elements,
        BLOCK_SIZE,
        1e-5,
    )
    return out


def pattern(in_4, in_0, in_1, in_2, in_3):
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace=True)
    return (tmp_7, )

def replacement_args(in_4, in_0, in_1, in_2, in_3):
    return (in_4, in_0, in_1, in_2, in_3)

def replacement_func():
    return fused_avgpool_bn_silu