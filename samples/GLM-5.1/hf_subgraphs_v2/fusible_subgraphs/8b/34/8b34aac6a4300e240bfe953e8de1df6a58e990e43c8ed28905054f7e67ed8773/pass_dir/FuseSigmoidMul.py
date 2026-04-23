import torch
import triton
import triton.language as tl

# Pattern: sigmoid(x) * y
# This matches both instances in the model

def pattern(x, y):
    sig = torch.sigmoid(x)
    result = y * sig
    return result

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_sigmoid_mul_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Cast to float32 for sigmoid (required for fp16/bf16)
    x_f32 = x.to(tl.float32)
    y_f32 = y.to(tl.float32)
    sig_x = tl.sigmoid(x_f32)
    out = y_f32 * sig_x
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_sigmoid_mul(x, y):
    out = torch.empty_like(y)
    n_elements = x.numel()
    BLOCK_SIZE = 2048
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_sigmoid_mul_kernel[grid](
        x_ptr=x, y_ptr=y, out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return fused_sigmoid_mul