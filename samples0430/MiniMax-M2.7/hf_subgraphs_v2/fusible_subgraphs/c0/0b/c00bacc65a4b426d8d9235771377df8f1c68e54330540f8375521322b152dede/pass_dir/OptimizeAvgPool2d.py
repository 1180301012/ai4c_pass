import torch
import triton
import triton.language as tl

def pattern(in_5):
    """
    Pattern: Average Pooling 2D
    This matches the pattern in model.py:
    tmp_7 = torch.nn.functional.avg_pool2d(in_5, 2, 2, 0, True, False, None)
    """
    tmp_7 = torch.nn.functional.avg_pool2d(in_5, 2, 2, 0, True, False, None)
    return tmp_7

def replacement_args(in_5):
    return (in_5,)

@triton.jit
def avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate output spatial dimensions
    out_H = (H + 2 * padding - kernel_size) // stride + 1
    out_W = (W + 2 * padding - kernel_size) // stride + 1
    
    pid = tl.program_id(0)
    num_blocks = N * C * out_H * out_W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    
    # Calculate output indices
    n = offsets // (C * out_H * out_W)
    c = (offsets // (out_H * out_W)) % C
    out_h = (offsets // out_W) % out_H
    out_w = offsets % out_W
    
    # Calculate input start position
    in_h_start = out_h * stride - padding
    in_w_start = out_w * stride - padding
    
    # Accumulate sum over kernel window
    acc = 0.0
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            in_h = in_h_start + kh
            in_w = in_w_start + kw
            # Check bounds (valid padding = 0)
            if 0 <= in_h < H and 0 <= in_w < W:
                in_idx = n * C * H * W + c * H * W + in_h * W + in_w
                acc += tl.load(x_ptr + in_idx)
    
    # Average
    out = acc / (kernel_size * kernel_size)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_avg_pool2d(x, kernel_size=2, stride=2, padding=0):
    """
    Optimized AvgPool2D using Triton kernel.
    
    Pooling parameters:
    - kernel_size: 2 (2x2 window)
    - stride: 2 (non-overlapping)
    - padding: 0 (no padding)
    """
    N, C, H, W = x.shape
    
    # Calculate output dimensions
    out_H = (H + 2 * padding - kernel_size) // stride + 1
    out_W = (W + 2 * padding - kernel_size) // stride + 1
    
    # Allocate output
    out = torch.empty(N, C, out_H, out_W, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    num_outputs = N * C * out_H * out_W
    BLOCK_SIZE = 1024
    num_programs = (num_outputs + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    avg_pool2d_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        N=N, C=C, H=H, W=W,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return triton_avg_pool2d