import torch
import triton
import triton.language as tl

@triton.jit
def fused_relu_flatten_kernel(
    x_ptr,
    out_ptr,
    B,
    C,
    H,
    W,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (B * C * H * W)
    
    size_hw = H * W
    size_chw = C * size_hw
    
    # Compute 4D indices
    b = offsets // size_chw
    c = (offsets % size_chw) // size_hw
    h = (offsets % size_hw) // W
    w = offsets % W
    
    # Compute input pointer offset
    x_ptr_offset = b * (C * H * W) + c * (H * W) + h * W + w
    x = tl.load(x_ptr + x_ptr_offset, mask=mask, other=0.0)
    
    # ReLU
    out = tl.max(x, 0.0)
    
    # Store to output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_flatten(x):
    B, C, H, W = x.shape
    output_size = B * C * H * W
    output = torch.empty((B, C * H * W), dtype=x.dtype, device=x.device)
    
    block_size = 1024
    grid_size = (output_size + block_size - 1) // block_size
    
    fused_relu_flatten_kernel[grid_size](
        x_ptr=x,
        out_ptr=output,
        B=B,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=block_size
    )
    return output

def pattern(x):
    relu_out = torch.nn.functional.relu(x)
    flatten_out = relu_out.flatten(1, -1)
    return flatten_out

def replacement_args(x):
    return (x,)

def replacement_func():
    return fused_relu_flatten