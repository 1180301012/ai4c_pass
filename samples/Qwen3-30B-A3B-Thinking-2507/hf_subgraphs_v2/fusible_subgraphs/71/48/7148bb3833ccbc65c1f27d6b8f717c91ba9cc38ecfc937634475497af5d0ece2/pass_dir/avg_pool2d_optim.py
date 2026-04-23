import torch
import triton
import triton.language as tl

def pattern(x):
    return torch.nn.functional.avg_pool2d(x, 3, 1, 1, False, False, None)

def replacement_args(x):
    return (x, )

@triton.jit
def avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    B, C, H, W,
    input_stride0, input_stride1, input_stride2, input_stride3,
    output_stride0, output_stride1, output_stride2, output_stride3,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    # Grid indices for the kernel
    b = tl.program_id(0)
    c = tl.program_id(1)
    h_block = tl.program_id(2)
    w_block = tl.program_id(3)
    # Calculate h and w indices for the block
    h_start = h_block * BLOCK_H
    w_start = w_block * BLOCK_W
    h_idx = h_start + tl.arange(0, BLOCK_H)
    w_idx = w_start + tl.arange(0, BLOCK_W)
    # Mask for valid h and w
    h_mask = h_idx < H
    w_mask = w_idx < W
    # Initialize sum and count for this block
    sum_val = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    count = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)
    # Iterate over 3x3 kernel
    for kh in range(-1, 2):
        for kw in range(-1, 2):
            # Input positions
            ih = h_idx + kh
            iw = w_idx + kw
            # Check if within input bounds
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            # Load input value
            input_val = tl.load(
                input_ptr +
                b * input_stride0 +
                c * input_stride1 +
                ih * input_stride2 +
                iw * input_stride3,
                mask=valid,
                other=0.0
            )
            sum_val += input_val
            count += valid
    # Calculate average and store
    average = sum_val / count
    tl.store(
        output_ptr +
        b * output_stride0 +
        c * output_stride1 +
        h_idx * output_stride2 +
        w_idx * output_stride3,
        average,
        mask=h_mask[:, None] & w_mask[None, :]
    )

@torch.fx.wrap
def avg_pool2d_torch(x):
    B, C, H, W = x.shape
    BLOCK_H = 16
    BLOCK_W = 16
    grid_h = (H + BLOCK_H - 1) // BLOCK_H
    grid_w = (W + BLOCK_W - 1) // BLOCK_W
    out = torch.empty_like(x)
    # Get strides
    input_strides = x.stride()
    output_strides = out.stride()
    avg_pool2d_kernel[(B, C, grid_h, grid_w), (BLOCK_H, BLOCK_W)](
        x,
        out,
        B, C, H, W,
        *input_strides,
        *output_strides,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W
    )
    return out

def replacement_func():
    return avg_pool2d_torch