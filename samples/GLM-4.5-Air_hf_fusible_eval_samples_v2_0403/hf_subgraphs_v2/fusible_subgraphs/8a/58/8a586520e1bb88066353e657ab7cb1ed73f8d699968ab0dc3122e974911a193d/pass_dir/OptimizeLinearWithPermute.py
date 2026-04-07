import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight, bias):
    linear = torch.nn.functional.linear(input_tensor, weight, bias)
    tmp_3 = linear.permute(0, 3, 1, 2)
    return tmp_3

def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)

@triton.jit
def linear_permute_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, seq_len1, seq_len2, in_features, out_features
):
    pid = tl.program_id(0)
    
    # Simple mapping: process one output element per program
    # Output is [batch, out_features, seq_len1, seq_len2]
    batch_idx = pid // (out_features * seq_len1 * seq_len2)
    out_idx = (pid % (out_features * seq_len1 * seq_len2)) // (seq_len1 * seq_len2)
    seq1_idx = (pid % (seq_len1 * seq_len2)) // seq_len2
    seq2_idx = pid % seq_len2
    
    # Check bounds
    if batch_idx >= batch_size:
        return
    if out_idx >= out_features:
        return
    if seq1_idx >= seq_len1:
        return
    if seq2_idx >= seq_len2:
        return
    
    # Load bias for this output feature
    bias_val = tl.load(bias_ptr + out_idx).to(tl.float32)
    
    # Compute linear operation
    input_base = input_ptr + (batch_idx * seq_len1 * seq_len2 * in_features + 
                             seq1_idx * seq_len2 * in_features + 
                             seq2_idx * in_features)
    
    acc = bias_val
    for k in range(in_features):
        input_val = tl.load(input_base + k).to(tl.float32)
        weight_val = tl.load(weight_ptr + out_idx * in_features + k).to(tl.float32)
        acc += input_val * weight_val
    
    # Store in permuted output location
    output_offset = batch_idx * out_features * seq_len1 * seq_len2 + \
                   out_idx * seq_len1 * seq_len2 + \
                   seq1_idx * seq_len2 + seq2_idx
    tl.store(output_ptr + output_offset, acc.to(tl.float16))

@torch.fx.wrap
def optimized_linear_permute(input_tensor, weight, bias):
    batch_size = input_tensor.shape[0]
    seq_len1 = input_tensor.shape[1]
    seq_len2 = input_tensor.shape[2]
    in_features = input_tensor.shape[3]
    
    # The output features come from weight.shape[0] - weight is [out_features, in_features]
    out_features = weight.shape[0]
    
    # Create output tensor with permuted shape [batch, out_features, seq_len1, seq_len2]
    output = torch.empty((batch_size, out_features, seq_len1, seq_len2), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Use one program per output element 
    total_elements = batch_size * out_features * seq_len1 * seq_len2
    
    # Launch kernel with 1D grid
    linear_permute_kernel[(total_elements,)](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        seq_len1=seq_len1,
        seq_len2=seq_len2,
        in_features=in_features,
        out_features=out_features
    )
    
    return output

def replacement_func():
    return optimized_linear_permute