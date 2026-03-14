import torch
import triton
import triton.language as tl

# Simple add pattern - the only one that matches
def pattern(in_0, in_1):
    return in_0 + in_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def triton_add_broadcast_kernel_p00(
    x_ptr,
    y_ptr,
    out_ptr,
    B, H, S,
    stride_x0, stride_x1, stride_x2, stride_x3,
    stride_y0, stride_y3,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    b = row_idx // (H * S)
    rem = row_idx % (H * S)
    h = rem // S
    s = rem % S
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < S
    
    x_offset = b * stride_x0 + h * stride_x1 + s * stride_x2 + col_offsets * stride_x3
    x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
    
    y_offset = b * stride_y0 + col_offsets * stride_y3
    y = tl.load(y_ptr + y_offset, mask=mask, other=0.0)
    
    out = x + y
    
    tl.store(out_ptr + x_offset, out, mask=mask)


@torch.fx.wrap
def triton_add_p00(x, y):
    B, H, S, S2 = x.shape
    
    out = torch.empty_like(x)
    n_rows = B * H * S
    
    BLOCK_SIZE = triton.next_power_of_2(S)
    if BLOCK_SIZE < 64:
        BLOCK_SIZE = 64
    
    triton_add_broadcast_kernel_p00[(n_rows,)](
        x, y, out,
        B, H, S,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return triton_add_p00