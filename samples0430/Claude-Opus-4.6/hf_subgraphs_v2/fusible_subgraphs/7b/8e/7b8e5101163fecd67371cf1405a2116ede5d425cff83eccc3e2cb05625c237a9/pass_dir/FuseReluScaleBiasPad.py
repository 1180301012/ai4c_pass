import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=4),
    ],
    key=['hw_out'],
)
@triton.jit
def fused_relu_scale_bias_pad_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    H,
    W,
    W_out,
    hw_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tl.program_id(1)
    pid_block = tl.program_id(0)

    # Load scalars once
    scale = tl.load(in_1_ptr)
    bias = tl.load(in_0_ptr)

    # Compute output element offsets for this block
    block_start = pid_block * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hw_out

    # Convert linear offset to 2D (h, w) in output space
    h_out = offsets // W_out
    w_out = offsets % W_out

    # Check if within original input bounds (padding is at bottom and right)
    in_bounds = (h_out < H) & (w_out < W)

    # Load input values where in bounds
    in_offset = pid_nc * (H * W) + h_out * W + w_out
    x = tl.load(in_2_ptr + in_offset, mask=mask & in_bounds, other=0.0)

    # Fused: ReLU + scale + bias
    x = tl.maximum(x, 0.0)
    x = x * scale + bias

    # Zero out padding positions
    x = tl.where(in_bounds, x, 0.0)

    # Store to output
    out_offset = pid_nc * hw_out + offsets
    tl.store(out_ptr + out_offset, x, mask=mask)


@torch.fx.wrap
def fused_relu_scale_bias_pad(in_0, in_1, in_2):
    N, C, H, W = in_2.shape
    H_out = H + 1
    W_out = W + 1

    out = torch.empty(N, C, H_out, W_out, dtype=in_2.dtype, device=in_2.device)

    N_C = N * C
    hw_out = H_out * W_out

    grid = lambda meta: ((hw_out + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'], N_C)

    fused_relu_scale_bias_pad_kernel[grid](
        in_0, in_1, in_2, out,
        H, W, W_out, hw_out,
    )

    return out


def replacement_func():
    return fused_relu_scale_bias_pad