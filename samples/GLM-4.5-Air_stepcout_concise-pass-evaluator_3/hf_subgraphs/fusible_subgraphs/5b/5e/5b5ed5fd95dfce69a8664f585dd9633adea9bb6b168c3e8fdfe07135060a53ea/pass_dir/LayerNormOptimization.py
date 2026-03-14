import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight, bias):
    # Layer normalization - simpler pattern without dynamic shape
    tmp_9 = torch.nn.functional.layer_norm(input_tensor, 512, weight, bias, 1e-06)
    return tmp_9

def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)

@triton.jit
def layer_norm_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, seq_len, hidden_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the output tensor
    pid = tl.program_id(0)
    
    # Calculate coordinates
    b = pid // (seq_len * hidden_size)
    s = (pid // hidden_size) % seq_len
    h = pid % hidden_size
    
    if b >= batch_size or s >= seq_len or h >= hidden_size:
        return
    
    # Calculate base index for this batch and sequence position
    batch_seq_idx = b * seq_len + s
    
    # Compute mean for the sequence position
    mean = 0.0
    for k in range(hidden_size):
        idx = batch_seq_idx * hidden_size + k
        val = tl.load(input_ptr + idx)
        mean += val
    mean /= hidden_size
    
    # Compute variance for the sequence position
    var = 0.0
    for k in range(hidden_size):
        idx = batch_seq_idx * hidden_size + k
        val = tl.load(input_ptr + idx)
        var += (val - mean) * (val - mean)
    var /= hidden_size
    var += eps
    std = tl.sqrt(var)
    
    # Load weight and bias for this hidden dimension
    w = tl.load(weight_ptr + h)
    bias_val = tl.load(bias_ptr + h)
    
    # Compute normalized output
    src_idx = batch_seq_idx * hidden_size + h
    x = tl.load(input_ptr + src_idx)
    normalized = (x - mean) / std
    result = normalized * w + bias_val
    
    tl.store(output_ptr + pid, result)

@torch.fx.wrap
def layer_norm_optimized(input_tensor, weight, bias):
    batch_size, seq_len, hidden_size = input_tensor.shape
    eps = 1e-06
    
    output = torch.empty_like(input_tensor)
    
    # Calculate total number of elements
    total_elements = batch_size * seq_len * hidden_size
    
    # Choose block size
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    layer_norm_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return layer_norm_optimized