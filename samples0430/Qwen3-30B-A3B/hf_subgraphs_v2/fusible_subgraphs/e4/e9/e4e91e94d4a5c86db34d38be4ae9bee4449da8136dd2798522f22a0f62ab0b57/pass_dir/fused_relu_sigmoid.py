import torch
import triton
import triton.language as tl

def pattern(x):
    relu_out = torch.nn.functional.relu(x, inplace=True)
    sig_out = torch.sigmoid(relu_out)
    return sig_out

def replacement_args(x):
    return (x,)

@triton.jit
def fused_relu_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, type=dtype)
    relu_x = tl.maximum(x, 0.0)
    sig_x = 1.0 / (1.0 + tl.exp(-relu_x))
    tl.store(out_ptr + offsets, sig_x, mask=mask)

@torch.fx.wrap
def fused_relu_sigmoid(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    if x.dtype == torch.float32:
        dtype = tl.float32
    elif x.dtype == torch.bfloat16:
        dtype = tl.bfloat16
    elif x.dtype == torch.float16:
        dtype = tl.float16
    else:
        raise RuntimeError(f"Unsupported dtype {x.dtype}")
    out = torch.empty_like(x)
    fused_relu_sigmoid_kernel[(num_blocks,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        dtype=dtype
    )
    return out

def replacement_func():
    return fused_relu_sigmoid