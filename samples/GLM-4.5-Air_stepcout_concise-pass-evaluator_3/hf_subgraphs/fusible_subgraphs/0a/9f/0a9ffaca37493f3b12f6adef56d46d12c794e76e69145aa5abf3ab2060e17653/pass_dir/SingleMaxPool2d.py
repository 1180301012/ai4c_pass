import torch
import triton
import triton.language as tl


# Pattern - match relu + max_pool2d chain
def pattern(in_0):
    x = torch.relu(in_0)
    out = torch.max_pool2d(x, 5, 1, 2, 1)
    return out


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


# Optimized Triton kernel for max_pool2d with kernel=5, stride=1, padding=2
@triton.jit
def max_pool2d_kernel(
    input_ptr, output_ptr,
    batch_stride, channel_stride, height_stride, width_stride,
    B, C, H, W, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_elements = B * C * H * W
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    b = offsets // (C * H * W)
    remaining = offsets % (C * H * W)
    c = remaining // (H * W)
    remaining = remaining % (H * W)
    h = remaining // W
    w = remaining % W
    
    input_offset = b * batch_stride + c * channel_stride + h * height_stride + w * width_stride
    
    x = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    x = tl.where(x > 0, x, 0.0)  # ReLU
    
    max_val = x
    
    for dh in range(-2, 3):
        for dw in range(-2, 3):
            if dh == 0 and dw == 0:
                continue
            h_neighbor = h + dh
            w_neighbor = w + dw
            
            valid_h = (h_neighbor >= 0) & (h_neighbor < H)
            valid_w = (w_neighbor >= 0) & (w_neighbor < W)
            valid = valid_h & valid_w
            
            if valid:
                neighbor_offset = b * batch_stride + c * channel_stride + h_neighbor * height_stride + w_neighbor * width_stride
                neighbor_val = tl.load(input_ptr + neighbor_offset, mask=mask, other=0.0)
                neighbor_val = tl.where(neighbor_val > 0, neighbor_val, 0.0)
                max_val = tl.where(max_val > neighbor_val, max_val, neighbor_val)
    
    tl.store(output_ptr + offsets, max_val, mask=mask)


@torch.fx.wrap
def max_pool2d_wrapper(in_0):
    B, C, H, W = in_0.shape
    device = in_0.device
    dtype = in_0.dtype
    
    out = torch.empty((B, C, H, W), device=device, dtype=dtype)
    
    BLOCK_SIZE = 1024
    num_elements = B * C * H * W
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    batch_stride = in_0.stride(0)
    channel_stride = in_0.stride(1)
    height_stride = in_0.stride(2)
    width_stride = in_0.stride(3)
    
    max_pool2d_kernel[(num_programs,)](
        in_0, out,
        batch_stride, channel_stride, height_stride, width_stride,
        B, C, H, W, BLOCK_SIZE
    )
    
    return out


def replacement_func():
    return max_pool2d_wrapper