import torch
import triton
import triton.language as tl


def pattern(conv_out, in_2):
    """
    Match: sigmoid(conv_out) * in_2 → hardtanh(0,6)
    conv_out: [B, C_out, 1, 1]   (output of preceding 1×1 conv)
    in_2:     [B, C_out, H, W]   (large feature map)
    """
    tmp_3 = conv_out.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5


def replacement_args(conv_out, in_2):
    return (conv_out, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=2, num_stages=3),
        triton.Config({'BLOCK_HW': 512},  num_warps=2, num_stages=3),
        triton.Config({'BLOCK_HW': 1024}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_HW': 2048}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 4096}, num_warps=4, num_stages=3),
    ],
    key=['C_OUT', 'HW'],
)
@triton.jit
def _sigmoid_mul_ht_kernel(
    scale_ptr,   # [B, C_OUT, 1, 1]  — sigmoid(conv2d output)
    in2_ptr,     # [B, C_OUT, H, W]  — feature map
    out_ptr,     # [B, C_OUT, H, W]  — output
    B, C_OUT, HW,
    BLOCK_HW: tl.constexpr,
):
    # Grid: (B * C_OUT, ceil(HW / BLOCK_HW))
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    b_idx = pid_bc // C_OUT
    c_idx = pid_bc  % C_OUT

    # Load the per-(b, c) scale value — one scalar load, reused for all HW elements
    scale = tl.load(scale_ptr + b_idx * C_OUT + c_idx).to(tl.float32)

    # Broadcast-multiply + ReLU6 (hardtanh[0,6])
    hw_off  = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_off < HW
    base    = b_idx * C_OUT * HW + c_idx * HW

    x   = tl.load(in2_ptr + base + hw_off, mask=hw_mask, other=0.0)
    out = x * scale
    out = tl.minimum(tl.maximum(out, 0.0), 6.0)
    tl.store(out_ptr + base + hw_off, out, mask=hw_mask)


@torch.fx.wrap
def _sigmoid_mul_ht_wrapper(conv_out, in_2):
    """
    conv_out : [B, C_out, 1, 1]
    in_2     : [B, C_out, H, W]
    """
    B     = in_2.shape[0]
    C_out = in_2.shape[1]
    H     = in_2.shape[2]
    W     = in_2.shape[3]
    HW    = H * W

    out = torch.empty_like(in_2)

    def grid(meta):
        return (B * C_out, (HW + meta['BLOCK_HW'] - 1) // meta['BLOCK_HW'])

    # Pass conv_out directly — its [B, C_out, 1, 1] contiguous layout is
    # identical to a flat [B*C_out] array; linear index b*C_out+c maps to
    # element [b, c, 0, 0] via ptr + b*C_out + c (strides: C_out, 1, 1, 1).
    _sigmoid_mul_ht_kernel[grid](
        conv_out, in_2, out,
        B, C_out, HW,
    )
    return out


def replacement_func():
    return _sigmoid_mul_ht_wrapper