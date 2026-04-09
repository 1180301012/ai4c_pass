import torch
import triton
import triton.language as tl

@triton.jit
def fused_first_element_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_sequences,
    hidden_size,
    output_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one sequence element
    seq_idx = tl.program_id(0)
    
    # Process only the first element of each sequence
    if seq_idx >= n_sequences:
        return
    
    # Load the entire first sequence element (first 384 elements)
    offset = seq_idx * hidden_size
    first_element = tl.load(input_ptr + offset + tl.arange(0, hidden_size))
    
    # Linear transformation: y = x @ W^T + b
    acc = tl.zeros((output_size,), dtype=tl.float32)
    
    # Vectorized matrix multiplication
    for k in range(0, hidden_size, BLOCK_SIZE):
        # Load input chunk
        input_chunk = first_element[k:k + BLOCK_SIZE]
        
        # Load corresponding weight rows from W^T (columns from W)
        weight_offset = k * output_size
        weight_block = tl.load(weight_ptr + weight_offset + tl.arange(0, min(BLOCK_SIZE, hidden_size - k)) * output_size)
        
        # Compute dot product
        acc += tl.sum(input_chunk[:, None] * weight_block, axis=0)
    
    # Add bias
    bias = tl.load(bias_ptr + tl.arange(0, output_size))
    result = acc + bias
    
    # Store result
    out_offset = seq_idx * output_size
    tl.store(out_ptr + out_offset, result)

@torch.fx.wrap
def fused_first_element_linear(input, weight, bias):
    import torch
    n_sequences = input.size(0)
    hidden_size = input.size(1)
    output_size = weight.size(0)
    
    out = torch.empty((n_sequences, output_size), dtype=input.dtype, device=input.device)
    
    BLOCK_SIZE = 128
    
    fused_first_element_linear_kernel[(n_sequences,)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_sequences=n_sequences,
        hidden_size=hidden_size,
        output_size=output_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(input, weight, bias):
    # Extract first element from each sequence
    first_element = input[(slice(None), 0)]
    # Apply linear transformation
    result = torch.nn.functional.linear(first_element, weight, bias)
    return result

def replacement_args(input, weight, bias):
    return (input, weight, bias)

def replacement_func():
    return fused_first_element_linear