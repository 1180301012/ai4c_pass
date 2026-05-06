import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    max_vals = torch.max(in_0, -1, keepdim=True)
    max_vals = max_vals[0]
    expanded_max = max_vals.expand_as(in_0)
    diff = expanded_max - in_0
    softmax_output = torch.nn.functional.softmax(diff, dim=-1)
    reshaped_in_1 = in_1.view(12, 512, -1)
    return (softmax_output, reshaped_in_1)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_softmax_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute softmax safely
    max_val = tl.max(x)
    x = x - max_val
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, dim=0)
    softmax_x = exp_x / sum_exp
    
    # Store results
    tl.store(output_ptr + offsets, softmax_x, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1):
    B, C, H = in_0.shape
    N = B * C * H
    
    input_flat = in_0.view(-1)
    output_flat = torch.empty_like(input_flat)
    
    optimized_softmax_kernel[(1,)](
        input_ptr=input_flat,
        output_ptr=output_flat,
        n_elements=N,
        BLOCK_SIZE=1024,
    )
    
    softmax_output = output_flat.view(B, C, H)
    reshaped_in_1 = in_1.view(12, 512, -1)
    return (softmax_output, reshaped_in_1)

def replacement_func():
    return kernel_wrapper