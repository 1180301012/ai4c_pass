import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'HW_BLOCK': 128}, num_warps=4),
        triton.Config({'HW_BLOCK': 256}, num_warps=4),
        triton.Config({'HW_BLOCK': 512}, num_warps=8),
        triton.Config({'HW_BLOCK': 1024}, num_warps=8),
    ],
    key=['C', 'H', 'W'],
)
@triton.jit
def fused_efficientformer_kernel(
    in0_ptr,    # [C] scale1 (per-channel scale)
    in2_ptr,    # [B, C, H, W] input feature map
    out_ptr,    # [B, C, H, W] output
    B, C, H, W,
    stride_b, stride_c, stride_h, stride_w,
    HW_BLOCK: tl.constexpr,
):
    """
    Fused kernel computing:
      relu_val = relu(in2)
      avg_val  = avg_pool2d(relu_val, kernel=3, stride=1, pad=1, count_include_pad=False)
      out      = relu_val + scale * (avg_val - relu_val)
    """
    pid_bc = tl.program_id(0)   # index over B*C
    pid_hw = tl.program_id(1)   # index over H*W blocks

    b_idx = pid_bc // C
    c_idx = pid_bc % C

    hw_start = pid_hw * HW_BLOCK
    hw_offsets = hw_start + tl.arange(0, HW_BLOCK)
    h_idx = hw_offsets // W
    w_idx = hw_offsets % W
    mask = hw_offsets < H * W

    # Load per-channel scale and upcast to float32
    scale = tl.load(in0_ptr + c_idx).to(tl.float32)

    # Base offset for this (batch, channel) slice
    base = b_idx * stride_b + c_idx * stride_c
    in_offsets = base + h_idx * stride_h + w_idx * stride_w

    # Load input values and apply ReLU (in float32 for accuracy)
    inp = tl.load(in2_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)
    relu_val = tl.maximum(inp, 0.0)

    # Compute avg_pool2d(3x3, stride=1, pad=1, count_include_pad=False)
    # For each output element, accumulate the 3x3 neighborhood of relu'd values.
    # Invalid (out-of-bounds) positions are excluded from sum AND count.
    pool_sum   = tl.zeros([HW_BLOCK], dtype=tl.float32)
    pool_count = tl.zeros([HW_BLOCK], dtype=tl.float32)

    for dh in range(-1, 2):
        for dw in range(-1, 2):
            nh = h_idx + dh
            nw = w_idx + dw
            valid = (nh >= 0) & (nh < H) & (nw >= 0) & (nw < W) & mask
            n_offsets = base + nh * stride_h + nw * stride_w
            # For masked-off positions, load returns other=0.0 → relu → 0.0 → no contribution
            n_val   = tl.load(in2_ptr + n_offsets, mask=valid, other=0.0).to(tl.float32)
            n_relu  = tl.maximum(n_val, 0.0)
            pool_sum   = pool_sum   + n_relu
            pool_count = pool_count + valid.to(tl.float32)

    avg_val = pool_sum / pool_count

    # Fused computation: relu + scale * (avg - relu)
    out_val = relu_val + scale * (avg_val - relu_val)

    # Store result (Triton auto-converts float32 → target dtype on store)
    tl.store(out_ptr + in_offsets, out_val, mask=mask)


@torch.fx.wrap
def fused_efficientformer(in_0, in_1, in_2):
    """
    Replacement for the fused subgraph:
      tmp_8  = relu(in_2) + in_0[:,None,None] * (avgpool2d(relu(in_2)) - relu(in_2))
      tmp_10 = in_1[:,None,None]   (zero-copy view)
    """
    B, C, H, W = in_2.shape

    out8 = torch.empty_like(in_2)

    grid = lambda meta: (B * C, triton.cdiv(H * W, meta['HW_BLOCK']))

    fused_efficientformer_kernel[grid](
        in_0, in_2, out8,
        B, C, H, W,
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
    )

    # tmp_10 = in_1.unsqueeze(-1).unsqueeze(-1) is a zero-copy view
    out10 = in_1.unsqueeze(-1).unsqueeze(-1)

    return (out8, out10)


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    tmp_2  = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3  = torch.nn.functional.avg_pool2d(tmp_2, 3, 1, 1, False, False, None)
    tmp_4  = tmp_3 - tmp_2
    tmp_5  = in_0.unsqueeze(-1)
    tmp_6  = tmp_5.unsqueeze(-1)
    tmp_7  = tmp_6 * tmp_4
    tmp_8  = tmp_2 + tmp_7
    tmp_9  = in_1.unsqueeze(-1)
    tmp_10 = tmp_9.unsqueeze(-1)
    return (tmp_8, tmp_10)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_efficientformer