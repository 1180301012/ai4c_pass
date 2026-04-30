import torch
import triton
import triton.language as tl


@triton.jit
def relu_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes sigmoid(relu(x)) in a single pass.
    This eliminates the need to materialize the intermediate relu output.
    """
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Step 1: ReLU - clip negative values to 0
    x_relu = tl.where(x > 0, x, 0.0)
    
    # Step 2: Sigmoid - 1 / (1 + exp(-x))
    # Use fast sigmoid approximation: x / (1 + |x|) is a common approximation
    # But for correctness, we use the full formula
    neg_x = -x_relu
    exp_neg_x = tl.exp(neg_x)
    out = 1.0 / (1.0 + exp_neg_x)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_relu_sigmoid(x):
    """Fused relu + sigmoid operation using Triton kernel."""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor with same dtype as input
    out = torch.empty_like(x)
    
    # Launch kernel
    relu_sigmoid_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0):
    """Match the relu -> sigmoid pattern."""
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = torch.sigmoid(tmp_0)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_relu_sigmoid