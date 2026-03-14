import torch
import triton
import triton.language as tl

def pattern(input, weight, bias, residual):
    # Simple pattern to match just linear operation and see what graph structure looks like
    tmp = torch.nn.functional.linear(input, weight, bias)
    return tmp

def replacement_args(input, weight, bias, residual):
    return (input, weight, bias)

@triton.jit
def diagnose_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, n_features, n_hidden,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size * n_hidden:
        return
    
    i = pid // n_hidden
    j = pid % n_hidden
    
    acc = 0.0
    for k in range(0, n_features, BLOCK_SIZE):
        k_end = min(k + BLOCK_SIZE, n_features)
        
        input_val = tl.load(input_ptr + i * n_features + k, 
                          mask=k < n_features, other=0.0)
        weight_val = tl.load(weight_ptr + j * n_features + k,
                           mask=k < n_features, other=0.0)
        
        acc += input_val * weight_val
    
    bias_val = tl.load(bias_ptr + j, mask=j < n_hidden, other=0.0)
    result = acc + bias_val
    
    tl.store(output_ptr + i * n_hidden + j, result)

@torch.fx.wrap
def diagnose_linear_func(input, weight, bias):
    batch_size = input.shape[0]
    n_features = input.shape[1]
    n_hidden = bias.shape[0]
    
    total_elements = batch_size * n_hidden
    num_programs = (total_elements + 1023) // 1024
    
    output = torch.empty((batch_size, n_hidden), dtype=torch.float32, device=input.device)
    
    diagnose_linear_kernel[(num_programs,)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        n_features=n_features,
        n_hidden=n_hidden,
        BLOCK_SIZE=32
    )
    
    return output

def replacement_func():
    return diagnose_linear_func