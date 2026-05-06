import torch
import triton
import triton.language as tl

def pattern(input, running_mean, running_var, weight, bias):
    return torch.nn.functional.batch_norm(
        input,
        running_mean,
        running_var,
        weight,
        bias,
        False,
        0.1,
        1e-05
    )

def replacement_args(input, running_mean, running_var, weight, bias):
    return (input, running_mean, running_var, weight, bias)

@triton.jit
def batch_norm_eval_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    eps: tl.float32 = 1e-05,
    BLOCK_SIZE: tl.constexpr = 1024
):
    # Each program handles a block of batch elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, B)
    
    # Process all channels in this block
    for c in tl.arange(0, C):
        # Get parameters for this channel
        running_mean_c = tl.load(running_mean_ptr + c)
        running_var_c = tl.load(running_var_ptr + c)
        weight_c = tl.load(weight_ptr + c)
        bias_c = tl.load(bias_ptr + c)
        
        # Process all batch elements in this channel
        for b in tl.arange(block_start, block_end):
            # Load input
            x = tl.load(input_ptr + (b * C) + c)
            
            # Compute batch norm output
            norm = (x - running_mean_c) / tl.sqrt(running_var_c + eps)
            out_val = norm * weight_c + bias_c
            
            # Store output
            tl.store(out_ptr + (b * C) + c, out_val)

@torch.fx.wrap
def batch_norm_eval_wrapper(input, running_mean, running_var, weight, bias):
    B, C = input.shape
    out = torch.empty_like(input)
    
    batch_norm_eval_kernel[tl.grid(tl.cimg_size(B, BLOCK_SIZE), 1)](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        B=B,
        C=C,
        eps=1e-05,
        BLOCK_SIZE=1024
    )
    return out

def replacement_func():
    return batch_norm_eval_wrapper