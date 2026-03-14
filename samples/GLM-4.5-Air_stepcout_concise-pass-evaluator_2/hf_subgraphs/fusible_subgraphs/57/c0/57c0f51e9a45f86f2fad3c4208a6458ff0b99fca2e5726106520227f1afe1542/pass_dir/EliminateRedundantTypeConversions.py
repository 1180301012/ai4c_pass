import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Original computation pattern
    in_1 += in_0
    tmp_0 = in_1
    tmp_1 = tmp_0.float()  # This conversion is redundant since input is already float32
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_1 = None
    tmp_3 = tmp_2.type_as(tmp_0)  # This conversion is also redundant
    tmp_2 = tmp_0 = None
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.1, training=False)
    tmp_3 = None
    return (tmp_4,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_add_softmax_dropout_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and bias
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Add bias element-wise
    added = x + bias
    
    # Compute softmax using Triton operations
    # For numerical stability, subtract max
    max_val = tl.max(added, mask=mask)
    added = added - max_val
    exp_val = tl.exp(added)
    sum_exp = tl.sum(exp_val, mask=mask)
    softmax = exp_val / sum_exp
    
    # Apply dropout (p=0.1 means keep 90% of values)
    mask_keep = tl.random(tl.program_id(0)) > 0.1
    dropout_result = softmax * mask_keep * (1.0 / 0.9)  # Scale for correct expected value
    
    # Store result
    tl.store(out_ptr + offsets, dropout_result, mask=mask)

@torch.fx.wrap
def optimized_add_softmax_dropout(x, bias):
    # Handle tensor shapes - operations are element-wise
    assert x.shape == bias.shape, "Input tensors must have the same shape"
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x, dtype=torch.float32)
    
    optimized_add_softmax_dropout_kernel[(num_programs,)](
        x_ptr=x,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_add_softmax_dropout