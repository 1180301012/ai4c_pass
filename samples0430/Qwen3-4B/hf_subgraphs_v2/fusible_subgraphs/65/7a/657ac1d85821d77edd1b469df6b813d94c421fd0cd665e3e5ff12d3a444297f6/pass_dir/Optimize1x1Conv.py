import torch
import triton
import triton.language as tl

def pattern(in_8, in_2, in_1):
    return torch.conv2d(in_8, in_2, in_1, (1, 1), (0, 0), (1, 1), 1)

def replacement_args(in_8, in_2, in_1):
    return (in_8, in_2, in_1)

def triton_conv1x1_kernel(x_ptr, w_ptr, out_ptr, batch_size, in_channels, out_channels, height, width, BLOCK_SIZE):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    for h in range(height):
        for w in range(width):
            for o in range(out_channels):
                val = 0.0
                for i in range(in_channels):
                    idx_x = (batch_size * in_channels + h * width + w) * in_channels + i
                    x_val = tl.load(x_ptr + idx_x)
                    w_val = tl.load(w_ptr + (o * in_channels + i))
                    val += x_val * w_val
                tl.store(out_ptr + (batch_size * out_channels + h * width + w) * out_channels + o, val)

def triton_conv1x1(x, w):
    batch_size, in_channels, height, width = x.shape
    out_channels = w.shape[0]
    
    grid_size = ( (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    out = torch.empty_like(x)
    
    triton_conv1x1_kernel[grid_size](
        x_ptr=x,
        w_ptr=w,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE=128,
    )
    return out

def replacement_func():
    return triton_conv1x1