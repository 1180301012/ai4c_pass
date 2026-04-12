import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Match the exact computation sequence from model.py
    in_1 += in_0
    in_2 = in_1
    tmp_1 = in_2.float()
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = tmp_2.type_as(in_2)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    return tmp_4

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def softmax_dropout_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    dropout_p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(x, axis=0)
    max_val = tl.broadcast_to(max_val, x.shape)
    
    # Compute softmax
    exp_x = tl.exp(x - max_val)
    sum_exp = tl.sum(exp_x, axis=0)
    sum_exp = tl.broadcast_to(sum_exp, exp_x.shape)
    softmax_out = exp_x / sum_exp
    
    # Apply dropout
    dropout_mask = tl.random(tl.arange(0, BLOCK_SIZE)) > (dropout_p * 32767)
    dropout_mask = tl.broadcast_to(dropout_mask, softmax_out.shape)
    result = softmax_out * dropout_mask / (1.0 - dropout_p)
    
    # Store output
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_softmax_dropout(x, p=0.1):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    softmax_dropout_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        dropout_p=p,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

@torch.fx.wrap  
def optimized_forward(in_0, in_1):
    # Add the two inputs
    result = in_1 + in_0
    
    # Apply fused softmax with dropout
    output = fused_softmax_dropout(result, p=0.1)
    
    return output

def replacement_func():
    return optimized_forward