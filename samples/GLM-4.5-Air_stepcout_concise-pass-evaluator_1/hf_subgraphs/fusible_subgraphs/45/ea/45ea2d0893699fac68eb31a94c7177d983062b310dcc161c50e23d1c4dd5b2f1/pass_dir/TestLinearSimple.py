import torch
import triton
import triton.language as tl

def pattern(input, weight, bias):
    # Just match linear operation to understand the graph structure
    result = torch.nn.functional.linear(input, weight, bias)
    return result

def replacement_args(input, weight, bias):
    return (input, weight, bias)

@triton.jit
def simple_linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, n_features, n_hidden,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one element
    pid = tl.program_id(0)
    
    # Calculate which element this program handles
    if pid >= batch_size * n_hidden:
        return
    
    # Convert to 2D coordinates
    i = pid // n_hidden
    j = pid % n_hidden
    
    # Linear operation: output = input @ weight^T + bias
    acc = 0.0
    for k in range(0, n_features, BLOCK_SIZE):
        k_end = min(k + BLOCK_SIZE, n_features)
        
        # Load input[i, k]
        input_val = tl.load(input_ptr + i * n_features + k, 
                          mask=k < n_features, other=0.0)
        # Load weight[j, k]
        weight_val = tl.load(weight_ptr + j * n_features + k,
                           mask=k < n_features, other=0.0)
        
        acc += input_val * weight_val
    
    # Add bias
    bias_val = tl.load(bias_ptr + j, mask=j < n_hidden, other=0.0)
    result = acc + bias_val
    
    # Store result
    tl.store(output_ptr + i * n_hidden + j, result)

@torch.fx.wrap
def simple_linear_func(input, weight, bias):
    batch_size = input.shape[0]
    n_features = input.shape[1]
    n_hidden = bias.shape[0]
    
    # Determine grid size
    total_elements = batch_size * n_hidden
    num_programs = (total_elements + 1023) // 1024  # 1024 programs max
    
    # Create output tensor
    output = torch.empty((batch_size, n_hidden), dtype=torch.float32, device=input.device)
    
    # Launch kernel
    simple_linear_kernel[(num_programs,)](
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
    return simple_linear_func