import torch
import triton
import triton.language as tl

def pattern(x):
    x_relu = torch.nn.functional.relu(x, inplace=True)
    x_dropout = torch.nn.functional.dropout2d(x_relu, 0.1, False, False)
    return x_dropout, x_relu
def replacement_args(x):
    return (x, )

@triton.jit
def fused_relu_dropout_kernel(
    x_ptr,
    out1_ptr,
    out2_ptr,
    n_elements: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE
    mask = (offsets + tl.arange(0, BLOCK_SIZE)) < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    relu = tl.where(x > 0, x, 0.0)
    dropout = relu * 0.9
    tl.store(out1_ptr + offsets, relu, mask=mask)
    tl.store(out2_ptr + offsets, dropout, mask=mask)

@torch.fx.wrap
def fused_relu_dropout(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out1 = torch.empty_like(x)
    out2 = torch.empty_like(x)
    fused_relu_dropout_kernel[(num_blocks,)](
        x_ptr=x,
        out1_ptr=out1,
        out2_ptr=out2,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return (out2, out1)

def replacement_func():
    return fused_relu_dropout