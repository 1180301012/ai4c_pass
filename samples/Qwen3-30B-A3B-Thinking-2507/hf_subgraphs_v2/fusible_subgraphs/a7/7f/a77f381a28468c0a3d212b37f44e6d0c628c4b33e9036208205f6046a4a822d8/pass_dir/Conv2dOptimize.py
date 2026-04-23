import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x, weight):
    conv = torch.conv2d(x, weight, None, (2, 2), (0, 0), (1, 1), 1)
    return conv

# Argument extraction function
def replacement_args(x, weight):
    return (x, weight, weight.shape[0])

# Optimized kernel
@triton.jit
def conv_kernel(
    x_ptr, x_stride0, x_stride1, x_stride2, x_stride3,
    weight_ptr, weight_stride0, weight_stride1,
    out_ptr, out_stride0, out_stride1, out_stride2, out_stride3,
    batch, in_ch, out_ch, height, width,
    BLOCK_SIZE: tl.constexpr
):
    # Get the current output channel index
    ch_idx = tl.program_id(1)
    if ch_idx >= out_ch:
        return

    # Compute spatial location (h, w)
    block_idx = tl.program_id(0)
    h = block_idx // width
    w = block_idx % width

    # Get base pointers
    x_base = x_ptr + h * x_stride2 + w * x_stride3
    out_base = out_ptr + h * out_stride2 + w * out_stride3

    # Compute dot product for this channel
    dot = 0.0
    for c in range(in_ch):
        x_val = tl.load(x_base + c * x_stride1)
        w_val = tl.load(weight_ptr + ch_idx * weight_stride0 + c * weight_stride1)
        dot += x_val * w_val

    tl.store(out_base + ch_idx * out_stride1, dot)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def conv_wrapper(x, weight, out_ch):
    # Extract dimensions
    batch, in_ch, height, width = x.shape

    # Output dimensions
    # Calculate output dimensions
    output_height = (height - 1) // 2 + 1
    output_width = (width - 1) // 2 + 1

    # Output dimensions
    out_shape = (batch, out_ch, output_height, output_width)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    # Calculate grid dimensions
    num_spatial_locs = output_height * output_width
    BLOCK_SIZE = 128  # Good for most hardware
    num_ch_blocks = (out_ch + BLOCK_SIZE - 1) // BLOCK_SIZE
    conv_kernel[(num_spatial_locs, num_ch_blocks)](
        x,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight,
        weight.stride(0), weight.stride(1),
        out,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch,
        in_ch,
        out_ch,
        height,
        width,
        BLOCK_SIZE
    )

    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return conv_wrapper