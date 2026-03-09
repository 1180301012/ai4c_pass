import torch
import triton
import triton.language as tl

def pattern(x):
    # Simple pattern with single argument to avoid dead code
    tmp_0 = torch.relu(x)
    tmp_1 = tmp_0 * 0.9
    return (tmp_1, tmp_0)

def replacement_args(x):
    return (x,)

@triton.jit
def fused_relu_dropout_kernel(
    x_ptr,
    relu_out_ptr,
    dropout_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute ReLU: max(x, 0)
    relu_out = tl.maximum(x, 0.0)
    
    # Apply dropout scaling (1 - p = 1 - 0.1 = 0.9)
    dropout_scale = 0.9
    dropout_out = relu_out * dropout_scale
    
    # Store both outputs
    tl.store(relu_out_ptr + offsets, relu_out, mask=mask)
    tl.store(dropout_out_ptr + offsets, dropout_out, mask=mask)

@torch.fx.wrap
def fused_relu_dropout(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    relu_out = torch.empty_like(x)
    dropout_out = torch.empty_like(x)
    
    fused_relu_dropout_kernel[(num_programs,)](
        x_ptr=x,
        relu_out_ptr=relu_out,
        dropout_out_ptr=dropout_out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return dropout_out, relu_out

def replacement_func():
    return fused_relu_dropout