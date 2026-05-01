import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    relu_out = torch.nn.functional.relu(in_2)
    mul_out = in_1 * relu_out
    add_out = mul_out + in_0
    pad_out = torch.nn.functional.pad(add_out, (0, 1, 0, 1), 'constant', None)
    return pad_out

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def fused_compute_kernel(
    in_ptr,
    out_ptr,
    scale,
    bias,
    N, C, H, W,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    h = tl.program_id(2) * BLOCK_H + tl.arange(0, BLOCK_H)
    w = tl.program_id(3) * BLOCK_W + tl.arange(0, BLOCK_W)

    # Mask for valid input region (h < H, w < W)
    mask = (h[:, None, None] < H) & (w[:, None, None] < W)

    # Compute input index
    in_idx = n[:, None, None] * (C * H * W) + c[None, :, None] * (H * W) + h[None, None, :] * W + w[None, None, :]
    in_val = tl.load(in_ptr + in_idx, mask=mask, other=0.0)

    # Apply operations
    relu_val = tl.maximum(in_val, 0.0)
    scaled_val = relu_val * scale
    final_val = scaled_val + bias
    out_val = tl.where(mask, final_val, 0.0)

    # Compute output index
    out_idx = n[:, None, None] * (C * (H + 1) * (W + 1)) + c[None, :, None] * ((H + 1) * (W + 1)) + h[None, None, :] * (W + 1) + w[None, None, :]
    tl.store(out_ptr + out_idx, out_val, mask=mask)

@torch.fx.wrap
def fused_compute(in_2, in_1, in_0):
    scale = in_1.item()
    bias = in_0.item()
    N, C, H, W = in_2.shape
    out = torch.empty((N, C, H + 1, W + 1), dtype=in_2.dtype, device=in_2.device)

    # Optimal block sizes (tuned for common GPU architectures)
    grid_n = (N + 31) // 32
    grid_c = (C + 7) // 8
    grid_h = (H + 1 + 15) // 16
    grid_w = (W + 1 + 15) // 16

    fused_compute_kernel[(grid_n, grid_c, grid_h, grid_w)](
        in_2,
        out,
        scale,
        bias,
        N, C, H, W,
        BLOCK_N=32,
        BLOCK_C=8,
        BLOCK_H=16,
        BLOCK_W=16
    )
    return out

def replacement_func():
    return fused_compute