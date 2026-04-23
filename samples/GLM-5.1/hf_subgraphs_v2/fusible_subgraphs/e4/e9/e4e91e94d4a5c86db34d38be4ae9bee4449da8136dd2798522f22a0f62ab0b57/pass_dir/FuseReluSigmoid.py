import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.sigmoid(tmp_0)
    return (tmp_1,)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_relu_sigmoid_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    # ReLU: max(0, x)
    relu_out = tl.maximum(x, 0.0)
    # Sigmoid: 1 / (1 + exp(-relu_out))
    # Use fast sigmoid: for positive values, sigmoid(x) = 1 / (1 + exp(-x))
    # since relu_out >= 0, we can compute directly
    neg_relu = -relu_out
    exp_neg = tl.exp(neg_relu)
    sigmoid_out = 1.0 / (1.0 + exp_neg)
    tl.store(out_ptr + offsets, sigmoid_out, mask=mask)

@torch.fx.wrap
def fused_relu_sigmoid(in_0):
    N = in_0.numel()
    BLOCK_SIZE = 512
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(in_0)
    fused_relu_sigmoid_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return (out,)

def replacement_func():
    return fused_relu_sigmoid