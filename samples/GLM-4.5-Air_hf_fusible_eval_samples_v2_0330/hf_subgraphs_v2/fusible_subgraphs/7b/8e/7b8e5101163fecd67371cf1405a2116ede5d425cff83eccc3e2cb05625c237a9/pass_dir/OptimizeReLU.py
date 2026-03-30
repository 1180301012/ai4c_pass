import torch
import triton
import triton.language as tl

def pattern(x):
    return torch.nn.functional.relu(x, inplace = False)

def replacement_args(x):
    return (x,)

@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * channels * height * width
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.math.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_relu(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    relu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        batch_size=x.shape[0],
        channels=x.shape[1] if len(x.shape) > 1 else 1,
        height=x.shape[2] if len(x.shape) > 2 else 1,
        width=x.shape[3] if len(x.shape) > 3 else 1,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_relu