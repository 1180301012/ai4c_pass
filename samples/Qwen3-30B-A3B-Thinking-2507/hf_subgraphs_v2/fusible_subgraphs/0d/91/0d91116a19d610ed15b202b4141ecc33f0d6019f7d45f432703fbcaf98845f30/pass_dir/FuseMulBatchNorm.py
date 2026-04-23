import torch
import triton
import triton.language as tl

@triton.jit
def fused_batchnorm_kernel(
    x_ptr,   # [B, C, H, W]
    s_ptr,   # [B, C, 1, 1] -> accessed as [B, C]
    scale_ptr,
    bias_shift_ptr,
    out_ptr,
    B: tl.int32,
    C: tl.int32,
    H: tl.int32,
    W: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    total_elements = B * C * H * W
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Convert 1D index to 4D indices: (b, c, h, w)
    b = offsets // (C * H * W)
    r = offsets % (C * H * W)
    c = r // (H * W)
    r = r % (H * W)
    h = r // W
    w = r % W

    # Load x
    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load s: s at (b, c)
    s_idx = b * C + c
    s_val = tl.load(s_ptr + s_idx, mask=mask, other=0.0)

    # Load scale and bias_shift for channel c
    scale_val = tl.load(scale_ptr + c, mask=mask, other=0.0)
    bias_shift_val = tl.load(bias_shift_ptr + c, mask=mask, other=0.0)

    # Compute
    temp = x_val * s_val
    temp = temp * scale_val
    out_val = temp + bias_shift_val

    tl.store(out_ptr + offsets, out_val, mask=mask)

@torch.fx.wrap
def fused_batchnorm(x, s, running_mean, running_var, bias, weight):
    # Precompute scale and bias_shift on CPU
    eps = 1e-05
    denom = torch.sqrt(running_var + eps)
    scale = weight / denom
    bias_shift = (-running_mean * weight) / denom + bias

    B, C, H, W = x.shape
    output = torch.empty_like(x)

    # Launch the Triton kernel
    block_size = 1024
    grid = (x.numel() + block_size - 1) // block_size

    fused_batchnorm_kernel[grid](
        x, s, scale, bias_shift, output,
        B, C, H, W,
        block_size
    )
    return output

def pattern(in_5, in_4, in_0, in_1, in_2, in_3):
    x = in_5 * in_4
    y = torch.nn.functional.batch_norm(x, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return y

def replacement_args(in_5, in_4, in_0, in_1, in_2, in_3):
    return (in_5, in_4, in_0, in_1, in_2, in_3)

def replacement_func():
    return fused_batchnorm